import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const server = join(repo, 'plugins/ruflo-metaharness/scripts/server.mjs');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
function wire(environment = {}) {
  const child = spawn(process.execPath, [server], { env: { PATH: process.env.PATH, ...environment }, stdio: ['pipe', 'pipe', 'pipe'] });
  let nextId = 0, buffer = '', stderr = '';
  const pending = new Map();
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdout.on('data', chunk => {
    buffer += chunk;
    while (buffer.includes('\n')) {
      const end = buffer.indexOf('\n'), value = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
      const waiter = pending.get(value.id);
      if (waiter) { pending.delete(value.id); clearTimeout(waiter.timer); waiter.accept(value); }
    }
  });
  child.on('exit', () => {
    for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(new Error(`MCP exited: ${stderr}`)); }
    pending.clear();
  });
  return {
    rpc(method, params) {
      return new Promise((accept, reject) => {
        const id = ++nextId, timer = setTimeout(() => { pending.delete(id); child.kill(); reject(new Error('GPU MCP fixture timed out')); }, 10000);
        pending.set(id, { accept, reject, timer });
        child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
      });
    },
    async close() { child.stdin.end(); if (child.exitCode === null) await new Promise(done => child.once('exit', done)); assert.equal(stderr, ''); },
  };
}
const errorCode = reply => JSON.parse(reply.result.content[0].text).code;

test('GPU MCP discovery is credential-free and tools reject model-supplied runtime overrides', async t => {
  const client = wire(); t.after(() => client.close());
  const listed = await client.rpc('tools/list', {});
  assert.deepEqual(listed.result.tools.map(tool => tool.name), ['experiment_run', 'gpu_loop_status', 'gpu_loop_tick']);
  for (const tool of listed.result.tools.filter(tool => tool.name.startsWith('gpu_loop_'))) {
    assert.equal(tool.inputSchema.additionalProperties, false);
    assert.deepEqual(tool.inputSchema.properties, {});
    const unavailable = await client.rpc('tools/call', { name: tool.name, arguments: {} });
    assert.equal(errorCode(unavailable), 'NOT_CONFIGURED');
    for (const key of ['configPath', 'adapter', 'secretRef', 'command', 'budgetUsd']) {
      const extra = await client.rpc('tools/call', { name: tool.name, arguments: { [key]: 'forbidden' } });
      assert.equal(errorCode(extra), 'INVALID_INPUT');
    }
  }
});

async function fixture(t, throwOnSubmit = false) {
  const root = await mkdtemp(join(tmpdir(), 'gpu-loop-mcp-')); t.after(() => rm(root, { recursive: true, force: true }));
  const stateRoot = join(root, 'state'); await mkdir(stateRoot, { mode: 0o700 });
  const marker = join(root, 'host-calls.json');
  const modulePath = join(root, 'host.mjs');
  const source = `import {writeFile} from 'node:fs/promises';
export async function submit({job,idempotencyKey}) {
  ${throwOnSubmit ? "throw new Error('DO-NOT-LEAK-SECRET-VALUE');" : `await writeFile(${JSON.stringify(marker)}, JSON.stringify({id:job.id,secretRef:job.secretRef,idempotencyKey}), {mode:0o600}); return {runId:'fixture-run'};`}
}
export async function inspect() { throw new Error('Unexpected inspect in first tick'); }
export async function cleanup() { return {released:true}; }
`;
  await writeFile(modulePath, source, { mode: 0o600 });
  const configPath = join(root, 'loop.json');
  await writeFile(configPath, JSON.stringify({ schema: 'ruvllm-gpu-loop/v1', stateRoot, intervalSeconds: 60,
    maxAttempts: 2, budgetUsd: 2, jobs: [{ id: 'fixture-eval', provider: 'vast.ai',
      model: { repo: 'Qwen/Qwen3.5-4B', revision: 'a'.repeat(40) }, task: 'evaluate',
      secretRef: 'ruos-vast-reference', resource: { instanceId: 'fixture-instance' }, maxCostUsd: 1,
      timeoutSeconds: 60 }] }), { mode: 0o600 });
  return { marker, configPath, environment: { RUVLLM_GPU_LOOP_CONFIG: configPath,
    RUVLLM_GPU_HOST_MODULE: modulePath, RUVLLM_GPU_HOST_SHA256: digest(source) } };
}

test('GPU MCP tick passes only opaque ruOS secret reference to pinned fixture host and persists status', async t => {
  const fixtureState = await fixture(t), client = wire(fixtureState.environment); t.after(() => client.close());
  const tick = await client.rpc('tools/call', { name: 'gpu_loop_tick', arguments: {} });
  assert.equal(tick.result.isError, false, JSON.stringify(tick));
  assert.equal(tick.result.structuredContent.jobs[0].status, 'running');
  const markerBefore = await readFile(fixtureState.marker);
  const submitted = JSON.parse(await readFile(fixtureState.marker, 'utf8'));
  assert.equal(submitted.secretRef, 'ruos-vast-reference');
  assert.equal(submitted.id, 'fixture-eval');
  assert.equal(typeof submitted.idempotencyKey, 'string');
  const repeated = await client.rpc('tools/call', { name: 'gpu_loop_tick', arguments: {} });
  assert.equal(repeated.result.isError, false);
  assert.deepEqual(await readFile(fixtureState.marker), markerBefore, 'A repeated immediate tick must not submit twice');
  const status = await client.rpc('tools/call', { name: 'gpu_loop_status', arguments: {} });
  assert.equal(status.result.isError, false, JSON.stringify(status));
  assert.deepEqual(JSON.parse(status.result.content[0].text), status.result.structuredContent);
  const hostless = wire({ RUVLLM_GPU_LOOP_CONFIG: fixtureState.configPath }); t.after(() => hostless.close());
  const readonly = await hostless.rpc('tools/call', { name: 'gpu_loop_status', arguments: {} });
  assert.equal(readonly.result.isError, false);
  const missingHost = await hostless.rpc('tools/call', { name: 'gpu_loop_tick', arguments: {} });
  assert.equal(errorCode(missingHost), 'NOT_CONFIGURED');
});

test('GPU MCP refuses changed adapter pins and never emits provider secret-bearing errors', async t => {
  const fixtureState = await fixture(t, true);
  const mismatched = wire({ ...fixtureState.environment, RUVLLM_GPU_HOST_SHA256: '0'.repeat(64) }); t.after(() => mismatched.close());
  const refused = await mismatched.rpc('tools/call', { name: 'gpu_loop_tick', arguments: {} });
  assert.equal(errorCode(refused), 'GPU_LOOP_FAILED');
  const client = wire(fixtureState.environment); t.after(() => client.close());
  const tick = await client.rpc('tools/call', { name: 'gpu_loop_tick', arguments: {} });
  assert(!JSON.stringify(tick).includes('DO-NOT-LEAK-SECRET-VALUE'));
  const status = await client.rpc('tools/call', { name: 'gpu_loop_status', arguments: {} });
  assert(!JSON.stringify(status).includes('DO-NOT-LEAK-SECRET-VALUE'));
  assert.equal(status.result.structuredContent.jobs[0].status, 'blocked');
});
