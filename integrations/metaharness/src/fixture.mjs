import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileDigest, sha256, checkedPath } from './engine.mjs';

// This deterministic fixture measures contracts and native I/O, never model quality.
export async function createFixture({ directory, binaryPath }) {
  directory = resolve(directory); binaryPath = await checkedPath(resolve(binaryPath));
  await mkdir(directory, { mode: 0o700 });
  const inputs = join(directory, 'inputs'), outputs = join(directory, 'outputs');
  await mkdir(inputs, { mode: 0o700 }); await mkdir(outputs, { mode: 0o700 });
  async function json(name, value) {
    const path = join(inputs, name), bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n');
    await writeFile(path, bytes, { flag: 'wx', mode: 0o600 }); return { path, sha256: sha256(bytes) };
  }
  const rawHeader = JSON.stringify({ 'layer.weight': { dtype: 'BF16', shape: [4], data_offsets: [0, 8] } });
  const header = Buffer.from(rawHeader + ' '.repeat((8 - Buffer.byteLength(rawHeader) % 8) % 8));
  const size = Buffer.alloc(8); size.writeBigUInt64LE(BigInt(header.length));
  const values = Buffer.alloc(8); [0x3f80, 0x4000, 0x4040, 0x4080].forEach((value, i) => values.writeUInt16LE(value, i * 2));
  const original = Buffer.concat([size, header, values]), source = join(inputs, 'original.safetensors');
  await writeFile(source, original, { flag: 'wx', mode: 0o600 });
  const candidate = Buffer.from(original); candidate.writeUInt16LE(0x4020, candidate.length - 6);
  const policy = await json('policy.json', { allowed_tensors: ['layer.weight'], max_changed_values: 1, max_l2: 0.5, max_abs_delta: 0.5, max_relative_l2: 1, protected: [] });
  const patch = await json('patch.json', { schema: 'ruvllm.weight_patch.v1', source_sha256: sha256(original), edits: [{ tensor: 'layer.weight', index: 1, old_bits: 0x4000, new_bits: 0x4020 }] });
  const binding = { runtime_sha256: await fileDigest(binaryPath), topology_sha256: '2'.repeat(64), protocol_sha256: '3'.repeat(64), toolchain_sha256: '4'.repeat(64) };
  const evaluation = await json('fixture-evaluation.json', {
    schema: 'ruvllm.weight_evaluation.v1',
    baseline: { ...binding, model_generation: 0, model_sha256: sha256(original), patch_sha256: '0'.repeat(64) },
    candidate: { ...binding, model_generation: 1, model_sha256: sha256(candidate), patch_sha256: patch.sha256 },
    suite_sha256: '5'.repeat(64), protocol_sha256: binding.protocol_sha256,
    baseline_output_sha256: '6'.repeat(64), candidate_output_sha256: '7'.repeat(64),
    evaluator_identity: 'synthetic-fixture-evaluator', proposer_identity: 'synthetic-fixture-proposer',
    total: 2, baseline_correct: 0, candidate_correct: 1, baseline_truncated: 1, candidate_truncated: 0,
  });
  const plan = await json('plan.json', { schema: 'ruvllm-experiment-plan/v1', evidenceMode: 'fixture', source, policy: policy.path, candidates: [{ id: 'fixture-improvement', patch: patch.path, evaluation: evaluation.path }, { id: 'unevaluated-control', patch: patch.path }] });
  const configPath = join(directory, 'runtime.json');
  await writeFile(configPath, JSON.stringify({ schema: 'ruvllm-experiment-runtime/v1', binaryPath, binarySha256: binding.runtime_sha256, readRoots: [inputs], writeRoot: outputs, policySha256s: [policy.sha256], maxCandidates: 16, timeoutMs: 30000 }, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  return { configPath, planPath: plan.path };
}
