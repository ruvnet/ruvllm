import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, realpath, chmod, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { tickLoop, readLoopStatus, readLoopConfig } from '../src/gpu-loop.mjs';

const revision = 'a'.repeat(40);
const job = id => ({ id, provider: 'vast.ai', model: { repo: 'Qwen/Qwen3.5-4B', revision }, task: 'evaluate',
  secretRef: 'ruos.vast-key', resource: { instanceId: 'existing-1' }, maxCostUsd: 1, timeoutSeconds: 60 });
const completion = (id, runId = 'run-1', idempotencyKey = 'old-key') => ({ completionVersion: 1, completionVerified: true, status: 'ok', exitCode: 0,
  assertionsVerified: true, evidenceComplete: true, truncated: false, jobId: id, modelRevision: revision,
  modelRepo: 'Qwen/Qwen3.5-4B', runId, idempotencyKey });
async function setup(t, overrides = {}) {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'gpu-loop-')));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const stateRoot = join(directory, 'state'); await mkdir(stateRoot, { mode: 0o700 });
  const config = { schema: 'ruvllm-gpu-loop/v1', stateRoot, intervalSeconds: 1, maxAttempts: 2, budgetUsd: 3, jobs: [job('evaluate-1')], ...overrides };
  const configPath = join(directory, 'config.json'); await writeFile(configPath, JSON.stringify(config), { mode: 0o600 });
  return { directory, stateRoot, configPath, config, now: 1000 };
}
function adapter(overrides = {}) {
  return { async submit() { return { runId: 'run-1' }; }, async inspect({ job, runId, idempotencyKey }) { return { status: 'completed', completion: completion(job.id, runId, idempotencyKey) }; },
    async cleanup() { return { released: true }; }, ...overrides };
}
test('scheduled host contract persists across restarts and succeeds only after cleanup', async t => {
  const f = await setup(t); const host = adapter();
  let state = await tickLoop({ ...f, adapter: host }); assert.equal(state.jobs[0].status, 'running');
  assert.equal(state.reservedUsd, 1); assert.equal(state.productionActivation, false);
  state = await tickLoop({ ...f, adapter: host, now: 1500 }); assert.equal(state.jobs[0].status, 'running');
  state = await tickLoop({ ...f, adapter: host, now: 2000 }); assert.equal(state.jobs[0].status, 'cleanup');
  state = await tickLoop({ ...f, adapter: host, now: 3000 }); assert.equal(state.jobs[0].status, 'succeeded');
  assert.match(state.jobs[0].history[0].completionSha256, /^[a-f0-9]{64}$/);
  assert.deepEqual(JSON.parse(await readFile(join(f.stateRoot, 'state.json'))), JSON.parse(JSON.stringify(await readLoopStatus(f))));
});
test('ambiguous submission blocks retries and redacts provider errors', async t => {
  const f = await setup(t); let calls = 0;
  const host = adapter({ async submit() { calls++; throw new Error('SECRET_VALUE_MUST_NOT_PERSIST'); } });
  const state = await tickLoop({ ...f, adapter: host }); assert.equal(state.jobs[0].status, 'blocked');
  await tickLoop({ ...f, adapter: host, now: 2000 }); assert.equal(calls, 1);
  assert(!JSON.stringify(state).includes('SECRET_VALUE')); assert.equal(state.reservedUsd, 1);
});
test('a persisted submitting intent is never automatically submitted again', async t => {
  const f = await setup(t); const state = await readLoopStatus(f); state.jobs[0] = { id: 'evaluate-1', status: 'submitting', attempts: 1, idempotencyKey: 'old-key', startedAt: 1 }; state.reservedUsd = 1;
  await writeFile(join(f.stateRoot, 'state.json'), JSON.stringify(state), { mode: 0o600 });
  let calls = 0; const result = await tickLoop({ ...f, adapter: adapter({ async submit() { calls++; } }) });
  assert.equal(result.jobs[0].status, 'blocked'); assert.equal(calls, 0);
});
test('cleanup must confirm release and blocks submission of the next job', async t => {
  const f = await setup(t, { jobs: [job('first'), job('second')] }); let submits = 0, cleanup = 0;
  const host = adapter({ async submit() { submits++; return { runId: 'run-1' }; }, async cleanup() { cleanup++; return { released: cleanup > 1 }; } });
  for (const now of [1000, 2000, 3000]) await tickLoop({ ...f, adapter: host, now });
  assert.equal(submits, 1); assert.equal((await readLoopStatus(f)).jobs[0].status, 'cleanup');
  await tickLoop({ ...f, adapter: host, now: 4000 }); await tickLoop({ ...f, adapter: host, now: 5000 }); assert.equal(submits, 2);
});
test('retry reserves another full ceiling and budget prevents another provider submission', async t => {
  const f = await setup(t, { budgetUsd: 1 }); let submits = 0;
  const host = adapter({ async submit() { submits++; return { runId: 'run-1' }; }, async inspect() { return { status: 'failed' }; } });
  for (const now of [1000, 2000, 3000, 4000]) await tickLoop({ ...f, adapter: host, now });
  const state = await readLoopStatus(f); assert.equal(submits, 1); assert.equal(state.jobs[0].status, 'failed'); assert.equal(state.reservedUsd, 1);
});
test('definite failure retries only after cleanup and stops at max attempts', async t => {
  const f = await setup(t); let submits = 0;
  const host = adapter({ async submit() { submits++; return { runId: `run-${submits}` }; }, async inspect() { return { status: 'failed' }; } });
  for (let now = 1000; now <= 8000; now += 1000) await tickLoop({ ...f, adapter: host, now });
  assert.equal(submits, 2); assert.equal((await readLoopStatus(f)).jobs[0].status, 'failed'); assert.equal((await readLoopStatus(f)).reservedUsd, 2);
  assert.equal((await readLoopStatus(f)).jobs[0].history.length, 2);
});
test('stdout-like or mismatched completions fail qualification and still release resources', async t => {
  for (const bad of [{ stdout: JSON.stringify(completion('evaluate-1')) }, { ...completion('evaluate-1'), modelRevision: 'b'.repeat(40) }, { ...completion('evaluate-1'), truncated: true }]) {
    const f = await setup(t); const host = adapter({ async inspect() { return { status: 'completed', completion: bad }; } });
    for (const now of [1000, 2000, 3000]) await tickLoop({ ...f, adapter: host, now });
    const state = await readLoopStatus(f); assert.equal(state.jobs[0].status, 'failed'); assert.equal(state.jobs[0].cleanupConfirmed, true);
  }
});
test('deadline triggers cleanup without declaring success', async t => {
  const f = await setup(t); let inspected = 0;
  const host = adapter({ async inspect() { inspected++; return { status: 'running' }; } });
  await tickLoop({ ...f, adapter: host }); await tickLoop({ ...f, adapter: host, now: 62000 });
  assert.equal(inspected, 0); assert.equal((await readLoopStatus(f)).jobs[0].status, 'cleanup');
  await tickLoop({ ...f, adapter: host, now: 63000 }); assert.equal((await readLoopStatus(f)).jobs[0].status, 'failed');
});
test('existing locks refuse overlap and are never auto-broken', async t => {
  const f = await setup(t); await mkdir(join(f.stateRoot, 'tick.lock'));
  await assert.rejects(tickLoop({ ...f, adapter: adapter() }), { code: 'EEXIST' });
  assert.equal((await readLoopStatus(f)).jobs[0].attempts, 0);
});
test('changed config cannot reuse bound state, and paths must be owner-private', async t => {
  const f = await setup(t); await tickLoop({ ...f, adapter: adapter() });
  await writeFile(f.configPath, JSON.stringify({ ...f.config, budgetUsd: 4 }));
  await assert.rejects(readLoopStatus(f), /binding/);
  await chmod(f.configPath, 0o644); await assert.rejects(readLoopStatus(f), /Owner-only/);
});
test('config rejects unknown fields, unpinned models, invalid budgets and symlinks', async t => {
  const f = await setup(t);
  for (const config of [{ ...f.config, rawSecret: 'forbidden' }, { ...f.config, budgetUsd: Infinity },
    { ...f.config, jobs: [{ ...job('test'), model: { repo: 'Qwen/model', revision: 'main' } }] }]) {
    await writeFile(f.configPath, JSON.stringify(config)); await assert.rejects(readLoopConfig(f));
  }
  const linked = join(f.directory, 'linked.json'); await symlink(f.configPath, linked);
  await assert.rejects(readLoopConfig({ configPath: linked }), /Symlinks/);
});
test('corrupt reservation ledger and missing running identity fail closed', async t => {
  const f = await setup(t); await tickLoop({ ...f, adapter: adapter() });
  const statePath = join(f.stateRoot, 'state.json'), original = JSON.parse(await readFile(statePath));
  await writeFile(statePath, JSON.stringify({ ...original, reservedUsd: 0 }));
  await assert.rejects(readLoopStatus(f), /ledger/);
  delete original.jobs[0].runId; await writeFile(statePath, JSON.stringify(original));
  await assert.rejects(readLoopStatus(f), /run identity/);
});
test('concurrent ticks cannot duplicate a provider submission', async t => {
  const f = await setup(t); let release, entered;
  const enteredPromise = new Promise(resolve => { entered = resolve; });
  const held = new Promise(resolve => { release = resolve; }); let calls = 0;
  const host = adapter({ async submit() { calls++; entered(); await held; return { runId: 'run-held' }; } });
  const first = tickLoop({ ...f, adapter: host }); await enteredPromise;
  await assert.rejects(tickLoop({ ...f, adapter: host }), { code: 'EEXIST' });
  release(); await first; assert.equal(calls, 1);
});
test('zero budget is disarmed and cannot submit provider work', async t => {
  const f = await setup(t, { budgetUsd: 0 }); let calls = 0;
  const state = await tickLoop({ ...f, adapter: adapter({ async submit() { calls++; } }) });
  assert.equal(calls, 0); assert.equal(state.jobs[0].status, 'failed'); assert.equal(state.reservedUsd, 0);
});
test('host receives an immutable job envelope and cannot accidentally change its budget or model', async t => {
  const f = await setup(t);
  const host = adapter({ async submit({ job }) {
    assert.equal(Object.isFrozen(job), true);
    assert.equal(Object.isFrozen(job.model), true);
    assert.throws(() => { job.maxCostUsd = 1000; }, TypeError);
    assert.throws(() => { job.model.revision = '0'.repeat(40); }, TypeError);
    return { runId: 'immutable-envelope-run' };
  } });
  const state = await tickLoop({ ...f, adapter: host });
  assert.equal(state.jobs[0].status, 'running');
  assert.equal(state.reservedUsd, f.config.jobs[0].maxCostUsd);
});
test('long schedule intervals wake at the job deadline and cleanup is immediately due', async t => {
  const f = await setup(t, { intervalSeconds: 86400 });
  const first = await tickLoop({ ...f, adapter: adapter() }); assert.equal(first.nextTickAt, 61000);
  const due = await tickLoop({ ...f, adapter: adapter(), now: 61000 }); assert.equal(due.jobs[0].status, 'cleanup'); assert.equal(due.nextTickAt, 61000);
  const done = await tickLoop({ ...f, adapter: adapter(), now: 61000 }); assert.equal(done.jobs[0].status, 'failed'); assert.equal(done.jobs[0].cleanupConfirmed, true);
});
test('old attempt completion cannot qualify a current provider attempt', async t => {
  const f = await setup(t);
  const host = adapter({ async inspect({ job }) { return { status: 'completed', completion: completion(job.id, 'old-run', 'old-key') }; } });
  await tickLoop({ ...f, adapter: host });
  const state = await tickLoop({ ...f, adapter: host, now: 2000 });
  assert.equal(state.jobs[0].status, 'cleanup'); assert.equal(state.jobs[0].targetState, 'failed'); assert.equal(state.jobs[0].reason, 'unverified_completion');
});
