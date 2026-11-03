import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { tickLoop } from '../src/gpu-loop.mjs';
import * as adapter from '../src/gpu-spool-adapter.mjs';

test('scheduled spool contract requires bound completion and explicit cleanup before success', async t => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'gpu-spool-loop-')));
  t.after(() => rm(root, { recursive: true, force: true }));
  const stateRoot = join(root, 'state'), spool = join(root, 'spool');
  await mkdir(stateRoot, { mode: 0o700 }); await mkdir(spool, { mode: 0o700 });
  const prior = process.env.RUVLLM_GPU_SPOOL_ROOT; process.env.RUVLLM_GPU_SPOOL_ROOT = spool;
  t.after(() => { if (prior === undefined) delete process.env.RUVLLM_GPU_SPOOL_ROOT; else process.env.RUVLLM_GPU_SPOOL_ROOT = prior; });
  const configPath = join(root, 'config.json');
  const job = { id: 'contract-evaluation', provider: 'vast.ai', task: 'evaluate', secretRef: 'ruos.example.vast',
    model: { repo: 'Qwen/Qwen3.5-4B', revision: 'a'.repeat(40) },
    resource: { instanceId: 'example-instance' }, maxCostUsd: 1, timeoutSeconds: 30 };
  await writeFile(configPath, JSON.stringify({ schema: 'ruvllm-gpu-loop/v1', stateRoot,
    intervalSeconds: 1, maxAttempts: 1, budgetUsd: 1, jobs: [job] }), { mode: 0o600 });
  const first = await tickLoop({ configPath, adapter, now: 1000 });
  const entry = first.jobs[0]; assert.equal(entry.status, 'running');
  assert.equal(JSON.parse(await readFile(join(spool, `${entry.runId}.request.json`))).job.secretRef, job.secretRef);
  await writeFile(join(spool, `${entry.runId}.status.json`), JSON.stringify({
    schema: 'ruvllm-gpu-spool-status/v1', runId: entry.runId, status: 'completed', costUsd: 0.1,
    completion: { completionVersion: 1, completionVerified: true, status: 'ok', exitCode: 0,
      assertionsVerified: true, evidenceComplete: true, truncated: false, jobId: job.id,
      modelRepo: job.model.repo, modelRevision: job.model.revision,
      runId: entry.runId, idempotencyKey: entry.idempotencyKey },
  }), { mode: 0o600 });
  const second = await tickLoop({ configPath, adapter, now: 2000 });
  assert.equal(second.jobs[0].status, 'cleanup');
  const pending = await tickLoop({ configPath, adapter, now: 2001 });
  assert.equal(pending.jobs[0].status, 'cleanup');
  const request = JSON.parse(await readFile(join(spool, `${entry.runId}.cleanup-request.json`)));
  assert.equal(request.jobId, job.id);
  await writeFile(join(spool, `${entry.runId}.cleanup.json`), JSON.stringify({
    schema: 'ruvllm-gpu-spool-cleanup/v1', runId: entry.runId, released: true,
  }), { mode: 0o600 });
  const final = await tickLoop({ configPath, adapter, now: 3001 });
  assert.equal(final.jobs[0].status, 'succeeded');
  assert.equal(final.reservedUsd, 1);
  assert.equal(final.productionActivation, false);
});
