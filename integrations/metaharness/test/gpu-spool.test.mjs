import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile, writeFile, realpath, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { submit, inspect, cleanup } from '../src/gpu-spool-adapter.mjs';

test('spool handoff stays pending until host supplies status and resource-release evidence', async t => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'gpu-spool-')));
  t.after(() => rm(root, { recursive: true, force: true }));
  const previous = process.env.RUVLLM_GPU_SPOOL_ROOT;
  process.env.RUVLLM_GPU_SPOOL_ROOT = root;
  t.after(() => { if (previous === undefined) delete process.env.RUVLLM_GPU_SPOOL_ROOT; else process.env.RUVLLM_GPU_SPOOL_ROOT = previous; });
  const job = { id: 'evaluate-4b', secretRef: 'ruos.vast', model: { revision: 'a'.repeat(40) } };
  const result = await submit({ job, idempotencyKey: 'attempt-1' });
  assert.equal(result.runId, 'attempt-1');
  assert.deepEqual(await submit({ job, idempotencyKey: 'attempt-1' }), result);
  const request = JSON.parse(await readFile(join(root, 'attempt-1.request.json')));
  assert.equal(request.job.secretRef, 'ruos.vast');
  assert.deepEqual(await inspect(result), { status: 'running' });
  assert.deepEqual(await cleanup({ ...result, job }), { released: false });
  await writeFile(join(root, 'attempt-1.status.json'), JSON.stringify({ schema: 'ruvllm-gpu-spool-status/v1',
    runId: 'attempt-1', status: 'completed', completion: { example: 'caller assertion only' } }), { mode: 0o600 });
  assert.equal((await inspect(result)).status, 'completed');
  await writeFile(join(root, 'attempt-1.cleanup.json'), JSON.stringify({ schema: 'ruvllm-gpu-spool-cleanup/v1',
    runId: 'attempt-1', released: true }), { mode: 0o600 });
  assert.deepEqual(await cleanup({ ...result, job }), { released: true });
  await assert.rejects(submit({ job: { ...job, id: 'other' }, idempotencyKey: 'attempt-1' }), /IDENTITY_CONFLICT/);
  await assert.rejects(inspect({ runId: '../escape' }), /INVALID_RUN_ID/);
});

test('spool refuses symlinked host replies and mismatched resource identity', async t => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'gpu-spool-boundary-')));
  t.after(() => rm(root, { recursive: true, force: true }));
  const previous = process.env.RUVLLM_GPU_SPOOL_ROOT; process.env.RUVLLM_GPU_SPOOL_ROOT = root;
  t.after(() => { if (previous === undefined) delete process.env.RUVLLM_GPU_SPOOL_ROOT; else process.env.RUVLLM_GPU_SPOOL_ROOT = previous; });
  await writeFile(join(root, 'other.json'), '{}', { mode: 0o600 });
  await symlink(join(root, 'other.json'), join(root, 'linked.status.json'));
  await assert.rejects(inspect({ runId: 'linked' }), /READ_FAILED/);
  await writeFile(join(root, 'wrong.status.json'), JSON.stringify({ schema: 'ruvllm-gpu-spool-status/v1',
    runId: 'different', status: 'completed' }), { mode: 0o600 });
  await assert.rejects(inspect({ runId: 'wrong' }), /INVALID_STATUS/);
});
