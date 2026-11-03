import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const plugin = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const binaryPath = process.env.RUFLO_NEURAL_TEST_BINARY;
const binarySha256 = process.env.RUFLO_NEURAL_TEST_BINARY_SHA256;
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const writeJson = (path, value) => writeFile(path, JSON.stringify(value), { mode: 0o600 });

function wire(configPath) {
  const env = { PATH: process.env.PATH, ...(configPath ? { RUFLO_NEURAL_CONFIG: configPath } : {}) };
  const child = spawn(process.execPath, [join(plugin, 'scripts/server.mjs')], { env, stdio: ['pipe', 'pipe', 'pipe'] });
  let sequence = 0; let stdout = ''; let stderr = ''; const pending = new Map();
  child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
    while (stdout.includes('\n')) {
      const newline = stdout.indexOf('\n'); const value = JSON.parse(stdout.slice(0, newline)); stdout = stdout.slice(newline + 1);
      const waiter = pending.get(value.id); if (waiter) { pending.delete(value.id); clearTimeout(waiter.timer); waiter.resolve(value); }
    }
  });
  child.on('exit', () => { for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(new Error(`MCP exited unexpectedly: ${stderr}`)); } pending.clear(); });
  function rpc(method, params) {
    return new Promise((resolveResult, reject) => {
      const id = ++sequence;
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`MCP timeout for ${method}`)); }, 10000);
      pending.set(id, { resolve: resolveResult, reject, timer });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }
  async function call(name, args, expectedError) {
    const response = await rpc('tools/call', { name, arguments: args }); assert.equal(response.error, undefined);
    if (expectedError) { assert.equal(response.result.isError, true); const err = JSON.parse(response.result.content[0].text); assert.equal(err.code, expectedError, JSON.stringify(response)); return err; }
    assert.equal(response.result.isError, false, JSON.stringify(response)); return response.result.structuredContent;
  }
  return { rpc, call, async close() { child.stdin.end(); if (child.exitCode === null) await new Promise((done) => child.once('exit', done)); assert.equal(stderr, ''); } };
}

async function fixture(t) {
  assert(binaryPath && /^[0-9a-f]{64}$/.test(binarySha256 ?? ''), 'Set RUFLO_NEURAL_TEST_BINARY and RUFLO_NEURAL_TEST_BINARY_SHA256 to the verified native executable');
  assert.equal(hash(await readFile(binaryPath)), binarySha256, 'Test binary pin differs');
  const root = await mkdtemp(join(tmpdir(), 'neural-plugin-wire-')); const inputs = join(root, 'inputs'); const output = join(root, 'outputs');
  await mkdir(inputs, { mode: 0o700 }); await mkdir(output, { mode: 0o700 });
  const checkpoint = join(inputs, 'base.safetensors');
  let header = JSON.stringify({ 'layer.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] } });
  header += ' '.repeat((8 - Buffer.byteLength(header) % 8) % 8);
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(Buffer.byteLength(header)));
  const scalars = Buffer.alloc(8); [0x3f80, 0x4000, 0x4040, 0x4080].forEach((bits, i) => scalars.writeUInt16LE(bits, i * 2));
  const original = Buffer.concat([length, Buffer.from(header), scalars]); await writeFile(checkpoint, original, { mode: 0o600 });
  const patch = join(inputs, 'patch.json'); const policy = join(inputs, 'policy.json'); const tensorList = join(inputs, 'tensors.json');
  await writeJson(patch, { schema: 'ruvllm.weight_patch.v1', source_sha256: hash(original), edits: [{ tensor: 'layer.weight', index: 1, old_bits: 0x4000, new_bits: 0x4040 }] });
  await writeJson(policy, { allowed_tensors: ['layer.weight'], max_changed_values: 2, max_l2: 2, max_abs_delta: 1, max_relative_l2: 0.2, protected: [{ tensor: 'layer.weight', start: 3, end: 4 }] });
  await writeJson(tensorList, ['layer.weight']);
  const configPath = join(root, 'runtime.json');
  const config = { schema: 'ruflo-neural-runtime/v1', binaryPath, binarySha256, readRoots: [inputs], writeRoot: output, policySha256s: [hash(await readFile(policy))], timeoutMs: 10000 };
  await writeJson(configPath, config);
  const client = wire(configPath);
  const initialized = await client.rpc('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'native-wire-test', version: '0.1.0' } });
  assert.equal(initialized.result.serverInfo.name, 'ruflo-neural-evolution');
  t.after(async () => { await client.close(); await rm(root, { recursive: true, force: true }); });
  return { root, inputs, output, checkpoint, original, patch, policy, tensorList, configPath, config, client };
}

test('real native MCP lifecycle inspects BF16, patches, validates, exports, verifies and restores exact bytes', async (t) => {
  const f = await fixture(t);
  const listed = await f.client.rpc('tools/list', {}); assert.equal(listed.result.tools.length, 6);
  assert.equal(listed.result.tools.some((tool) => /activate|authorize|approve/.test(tool.name)), false);
  const inspect = await f.client.call('weight_inspect', { source: f.checkpoint, tensorList: f.tensorList });
  assert.equal(inspect.result.source_sha256, hash(f.original)); assert.equal(inspect.result.activation_statistics_available, false);
  assert.equal(inspect.result.tensors[0].dtype, 'BF16'); assert.equal(inspect.result.tensors[0].nonfinite, 0);
  const candidatePath = join(f.output, 'candidate.safetensors'); const receiptPath = join(f.output, 'patch-receipt.json');
  const patched = await f.client.call('weight_patch', { source: f.checkpoint, patch: f.patch, policy: f.policy, output: candidatePath, receiptOutput: receiptPath });
  assert.equal(patched.activationAuthorized, false); assert.equal(patched.result.changed_values, 1);
  assert.equal(patched.result.candidate_sha256, hash(await readFile(candidatePath))); assert.deepEqual(await readFile(f.checkpoint), f.original);
  const d = hash('synthetic-test-fixture');
  const binding = (candidate) => ({ model_generation: Number(candidate), model_sha256: candidate ? patched.result.candidate_sha256 : hash(f.original), patch_sha256: candidate ? patched.result.patch_sha256 : '0'.repeat(64), runtime_sha256: binarySha256, topology_sha256: d, protocol_sha256: d, toolchain_sha256: d });
  const evaluationPath = join(f.inputs, 'synthetic-evaluation.json');
  await writeJson(evaluationPath, { schema: 'ruvllm.weight_evaluation.v1', baseline: binding(false), candidate: binding(true), suite_sha256: d, protocol_sha256: d, baseline_output_sha256: hash('synthetic-base'), candidate_output_sha256: hash('synthetic-candidate'), evaluator_identity: 'fixture-evaluator', proposer_identity: 'fixture-proposer', total: 100, baseline_correct: 40, candidate_correct: 90, baseline_truncated: 2, candidate_truncated: 0 });
  const evaluation = await f.client.call('weight_evaluate', { evaluation: evaluationPath, receipt: receiptPath }); assert.equal(evaluation.result.valid, true); assert.equal(evaluation.result.activation_authorized, false);
  const artifact = join(f.output, 'experiment.rvf');
  const exported = await f.client.call('weight_export', { patch: f.patch, policy: f.policy, receipt: receiptPath, evaluation: evaluationPath, output: artifact });
  assert.equal(exported.result.integrity_verified, true); assert.equal(exported.result.activation_authorized, false);
  const expected = join(f.inputs, 'rvf-digest.json'); await writeJson(expected, { sha256: exported.result.sha256 });
  const verified = await f.client.call('weight_verify', { artifact, expectedDigestFile: expected }); assert.equal(verified.result.integrity_verified, true);
  const restored = join(f.output, 'restored.safetensors'); const rollbackReceipt = join(f.output, 'rollback-receipt.json');
  const rolled = await f.client.call('weight_rollback', { source: candidatePath, output: restored, patch: f.patch, policy: f.policy, receipt: receiptPath, receiptOutput: rollbackReceipt });
  assert.equal(rolled.result.candidate_sha256, hash(f.original)); assert.deepEqual(await readFile(restored), f.original);
  await f.client.call('weight_patch', { source: f.checkpoint, patch: f.patch, policy: f.policy, output: candidatePath, receiptOutput: receiptPath }, 'OUTPUT_EXISTS');
});

test('wire boundary rejects escape, unknown operations, policy substitution and native drift', async (t) => {
  const f = await fixture(t);
  const outside = join(f.root, 'outside.safetensors'); await writeFile(outside, f.original);
  await f.client.call('weight_inspect', { source: join(f.inputs, '..', 'outside.safetensors'), tensorList: f.tensorList }, 'PATH_DENIED');
  const linked = join(f.inputs, 'escape.safetensors'); await symlink(outside, linked);
  await f.client.call('weight_inspect', { source: linked, tensorList: f.tensorList }, 'PATH_DENIED');
  const patchArgs = { source: f.checkpoint, patch: f.patch, policy: f.policy, output: join(f.root, 'outside-output.safetensors'), receiptOutput: join(f.output, 'receipt.json') };
  await f.client.call('weight_patch', patchArgs, 'PATH_DENIED');
  await f.client.call('weight_activate', {}, 'UNKNOWN_TOOL');
  await f.client.call('weight_inspect', { source: f.checkpoint, tensorList: f.tensorList, command: '/bin/sh' }, 'INVALID_INPUT');
  const permissive = join(f.inputs, 'unapproved-policy.json'); await writeJson(permissive, { allowed_tensors: ['layer.weight'], max_changed_values: 100, max_l2: 100, max_abs_delta: 100 });
  await f.client.call('weight_patch', { ...patchArgs, output: join(f.output, 'candidate.safetensors'), policy: permissive }, 'POLICY_DENIED');
  await writeJson(f.configPath, { ...f.config, binarySha256: '0'.repeat(64) });
  await f.client.call('weight_inspect', { source: f.checkpoint, tensorList: f.tensorList }, 'BINARY_DRIFT');
  await writeJson(f.configPath, f.config);
  const first = f.client.call('weight_inspect', { source: f.checkpoint, tensorList: f.tensorList });
  await f.client.call('weight_inspect', { source: f.checkpoint, tensorList: f.tensorList }, 'BUSY');
  assert.equal((await first).result.source_sha256, hash(f.original));
});

test('SDK request metadata is accepted without widening native arguments or path authority', async (t) => {
  const f = await fixture(t);
  const args = { source: f.checkpoint, tensorList: f.tensorList };
  for (const progressToken of [3, 'claude-tool-progress']) {
    // MCP Request.params permits _meta beside name/arguments. The SDK may add
    // progressToken and client extensions even when the tool schema has no meta.
    const response = await f.client.rpc('tools/call', { name: 'weight_inspect', arguments: args,
      _meta: { progressToken, 'client.example/context': { trace: 'fixture-only' },
        command: '/bin/sh', source: '/outside/not-a-native-input' } });
    assert.equal(response.error, undefined);
    assert.equal(response.result.isError, false);
    assert.equal(response.result.structuredContent.result.source_sha256, hash(f.original));
  }
  for (const metadata of [null, [], 'invalid', { progressToken: true }, { progressToken: {} }]) {
    const response = await f.client.rpc('tools/call', { name: 'weight_inspect', arguments: args, _meta: metadata });
    assert.equal(response.error?.code, -32600);
  }
  const unknown = await f.client.rpc('tools/call', { name: 'weight_inspect', arguments: args, command: '/bin/sh', _meta: {} });
  assert.equal(unknown.error?.code, -32600);
  const nested = await f.client.rpc('tools/call', { name: 'weight_inspect', arguments: { ...args, _meta: {} }, _meta: {} });
  assert.equal(nested.result.isError, true);
  assert.equal(JSON.parse(nested.result.content[0].text).code, 'INVALID_INPUT');
  const outside = join(f.root, 'outside.safetensors'); await writeFile(outside, f.original);
  const denied = await f.client.rpc('tools/call', { name: 'weight_inspect', arguments: { ...args, source: outside }, _meta: { approved: true } });
  assert.equal(JSON.parse(denied.result.content[0].text).code, 'PATH_DENIED');
});

test('MCP discovery works but every tool refuses an unconfigured runtime', async () => {
  const client = wire();
  try {
    await client.rpc('initialize', { protocolVersion: '2025-06-18' });
    assert.equal((await client.rpc('tools/list', {})).result.tools.length, 6);
    await client.call('weight_inspect', { source: '/tmp/unknown', tensorList: '/tmp/unknown' }, 'NOT_CONFIGURED');
  } finally { await client.close(); }
});
