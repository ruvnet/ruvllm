// Persistent orchestration only. A trusted host adapter owns provider APIs and secret resolution.
import { randomUUID, createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { open, lstat, mkdir, rename, rm } from 'node:fs/promises';
import { join, isAbsolute, resolve } from 'node:path';
import { checkedPath, parseJson } from './engine.mjs';

const fail = (ok, message) => { if (!ok) throw new Error(message); };
const id = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const amount = value => typeof value === 'number' && Number.isFinite(value) && value >= 0
  && Number.isSafeInteger(Math.round(value * 1e6)) && Math.round(value * 1e6) / 1e6 === value;
const exact = (value, required, optional = []) => fail(value && typeof value === 'object' && !Array.isArray(value)
  && required.every(key => Object.hasOwn(value, key)) && Object.keys(value).every(key => [...required, ...optional].includes(key)), 'Invalid or unknown fields');
const privateOwner = stat => fail((stat.mode & 0o077) === 0 && stat.uid === process.getuid(), 'Owner-only paths required');
const freeze = value => {
  if (value && typeof value === 'object') {
    for (const nested of Object.values(value)) freeze(nested);
    Object.freeze(value);
  }
  return value;
};

async function readJson(path) {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = await handle.stat(); privateOwner(stat);
    fail(stat.isFile() && stat.size <= 1024 * 1024, 'Bounded regular JSON file required');
    const bytes = await handle.readFile(); fail(bytes.length === stat.size, 'JSON file changed');
    return { value: parseJson(bytes), digest: sha(bytes) };
  } finally { await handle.close(); }
}

export async function readLoopConfig({ configPath }) {
  await checkedPath(configPath);
  const { value: config, digest } = await readJson(configPath);
  exact(config, ['schema', 'stateRoot', 'intervalSeconds', 'maxAttempts', 'budgetUsd', 'jobs']);
  fail(config.schema === 'ruvllm-gpu-loop/v1', 'Unsupported loop config');
  fail(Number.isInteger(config.intervalSeconds) && config.intervalSeconds >= 1 && config.intervalSeconds <= 86400, 'Invalid interval');
  fail(Number.isInteger(config.maxAttempts) && config.maxAttempts >= 1 && config.maxAttempts <= 10, 'Invalid attempts');
  fail(amount(config.budgetUsd), 'Invalid loop budget');
  await checkedPath(config.stateRoot, { directory: true }); privateOwner(await lstat(config.stateRoot));
  fail(configPath !== join(config.stateRoot, 'state.json') && configPath !== join(config.stateRoot, 'tick.lock'), 'Config collides with scheduler state');
  fail(Array.isArray(config.jobs) && config.jobs.length >= 1 && config.jobs.length <= 128, 'Expected 1..128 jobs');
  const seen = new Set();
  for (const job of config.jobs) {
    exact(job, ['id', 'provider', 'model', 'task', 'secretRef', 'resource', 'maxCostUsd', 'timeoutSeconds'], ['experiment']);
    fail(id(job.id) && !seen.has(job.id), 'Invalid or duplicate job ID'); seen.add(job.id);
    fail(job.provider === 'vast.ai' && ['evaluate', 'train'].includes(job.task), 'Unsupported provider/task');
    exact(job.model, ['repo', 'revision']);
    fail(typeof job.model.repo === 'string' && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(job.model.repo)
      && /^[a-f0-9]{40}$/.test(job.model.revision), 'Pinned model revision required');
    fail(id(job.secretRef), 'Opaque host secret reference required');
    exact(job.resource, [], ['instanceId', 'offerId']);
    fail(Object.keys(job.resource).length === 1 && Object.values(job.resource).every(id), 'Exactly one instanceId or offerId required');
    fail(amount(job.maxCostUsd) && job.maxCostUsd > 0, 'Invalid job cost ceiling');
    fail(Number.isInteger(job.timeoutSeconds) && job.timeoutSeconds >= 1 && job.timeoutSeconds <= 86400, 'Invalid job timeout');
    if (job.experiment) {
      exact(job.experiment, ['configPath', 'planPath']);
      fail(Object.values(job.experiment).every(path => typeof path === 'string' && isAbsolute(path) && path === resolve(path)), 'Absolute experiment paths required');
    }
  }
  return freeze({ ...config, configSha256: digest });
}

function initial(config) {
  return { schema: 'ruvllm-gpu-loop-state/v1', configSha256: config.configSha256,
    nextTickAt: 0, reservedUsd: 0, jobs: config.jobs.map(job => ({ id: job.id, status: 'queued', attempts: 0 })), productionActivation: false };
}
async function load(config) {
  const path = join(config.stateRoot, 'state.json');
  let state;
  try { state = (await readJson(path)).value; }
  catch (error) { if (error.code === 'ENOENT') return initial(config); throw error; }
  fail(state.schema === 'ruvllm-gpu-loop-state/v1' && state.configSha256 === config.configSha256, 'State/config binding mismatch');
  fail(Array.isArray(state.jobs) && state.jobs.length === config.jobs.length && state.jobs.every((job, index) =>
    job.id === config.jobs[index].id && ['queued', 'submitting', 'running', 'cleanup', 'succeeded', 'failed', 'blocked'].includes(job.status)
    && Number.isInteger(job.attempts) && job.attempts >= 0 && job.attempts <= config.maxAttempts), 'Invalid state records');
  fail(amount(state.reservedUsd) && state.reservedUsd <= config.budgetUsd
    && Number.isSafeInteger(state.nextTickAt) && state.nextTickAt >= 0 && state.productionActivation === false, 'Invalid state budget/timing');
  fail(Math.round(state.reservedUsd * 1e6) === state.jobs.reduce((sum, entry, index) =>
    sum + entry.attempts * Math.round(config.jobs[index].maxCostUsd * 1e6), 0), 'State reservation ledger mismatch');
  for (const entry of state.jobs) {
    if (['submitting', 'running', 'cleanup', 'succeeded', 'blocked'].includes(entry.status)) {
      fail(entry.attempts > 0 && id(entry.idempotencyKey) && Number.isSafeInteger(entry.startedAt) && entry.startedAt >= 0, 'Missing attempt identity');
    }
    if (['running', 'cleanup', 'succeeded'].includes(entry.status)) fail(id(entry.runId), 'Missing provider run identity');
    if (entry.status === 'cleanup') fail(['succeeded', 'failed'].includes(entry.targetState) && typeof entry.retryable === 'boolean', 'Invalid cleanup intent');
    if (entry.status === 'succeeded') fail(entry.cleanupConfirmed === true, 'Completion requires cleanup');
  }
  return state;
}
async function persist(config, state) {
  const temporary = join(config.stateRoot, `.state-${randomUUID()}.json`);
  const handle = await open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try { await handle.writeFile(JSON.stringify(state, null, 2) + '\n'); await handle.sync(); } finally { await handle.close(); }
  await rename(temporary, join(config.stateRoot, 'state.json'));
  const directory = await open(config.stateRoot, constants.O_RDONLY); try { await directory.sync(); } finally { await directory.close(); }
}
async function call(method, input, milliseconds) {
  const controller = new AbortController(); let timer;
  try {
    return await Promise.race([Promise.resolve().then(() => method({ ...input, signal: controller.signal })),
      new Promise((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new Error('ADAPTER_TIMEOUT')); }, milliseconds); })]);
  } finally { clearTimeout(timer); }
}
function verified(completion, job, entry) {
  return completion && completion.completionVersion === 1 && completion.completionVerified === true
    && completion.status === 'ok' && completion.exitCode === 0 && completion.assertionsVerified === true
    && completion.evidenceComplete === true && completion.truncated === false
    && completion.jobId === job.id && completion.modelRevision === job.model.revision
    && completion.modelRepo === job.model.repo && completion.runId === entry.runId
    && completion.idempotencyKey === entry.idempotencyKey;
}

export async function readLoopStatus({ configPath }) { return load(await readLoopConfig({ configPath })); }

export async function tickLoop({ configPath, adapter, now = Date.now() }) {
  fail(Number.isSafeInteger(now) && now >= 0, 'Invalid clock');
  fail(adapter && ['submit', 'inspect', 'cleanup'].every(method => typeof adapter[method] === 'function'), 'Host adapter required');
  const config = await readLoopConfig({ configPath }), lock = join(config.stateRoot, 'tick.lock');
  await mkdir(lock, { mode: 0o700 }); // Never break an existing lock: operator reconciliation is required after a crash.
  try {
    const state = await load(config);
    const deadlineDue = state.jobs.some(entry => entry.status === 'running'
      && now >= entry.startedAt + config.jobs.find(job => job.id === entry.id).timeoutSeconds * 1000);
    if (state.jobs.some(job => job.status === 'blocked') || (now < state.nextTickAt && !deadlineDue)) return state;
    state.nextTickAt = now + config.intervalSeconds * 1000;
    const entry = state.jobs.find(job => ['submitting', 'running', 'cleanup'].includes(job.status))
      ?? state.jobs.find(job => job.status === 'queued');
    if (!entry) { await persist(config, state); return state; }
    const job = config.jobs.find(job => job.id === entry.id);
    if (entry.status === 'submitting') { entry.status = 'blocked'; entry.reason = 'ambiguous_submission_reconcile'; }
    else if (entry.status === 'queued') {
      const reserved = Math.round(state.reservedUsd * 1e6) + Math.round(job.maxCostUsd * 1e6);
      if (entry.attempts >= config.maxAttempts || reserved > Math.round(config.budgetUsd * 1e6)) {
        entry.status = 'failed'; entry.reason = 'attempt_or_budget_limit';
      } else {
        entry.attempts++; state.reservedUsd = reserved / 1e6;
        for (const key of ['runId', 'cleanupConfirmed', 'targetState', 'retryable', 'completionSha256']) delete entry[key];
        entry.status = 'submitting'; entry.idempotencyKey = randomUUID(); entry.startedAt = now;
        await persist(config, state); // Persist intent and reservation before the provider effect.
        try {
          const submitted = await call(adapter.submit.bind(adapter), { job, idempotencyKey: entry.idempotencyKey }, Math.min(30000, job.timeoutSeconds * 1000));
          fail(submitted && id(submitted.runId), 'Invalid provider run ID');
          entry.runId = submitted.runId; entry.status = 'running'; entry.reason = 'submitted';
        } catch { entry.status = 'blocked'; entry.reason = 'ambiguous_submission_reconcile'; }
      }
    } else if (entry.status === 'running') {
      let result;
      if (now >= entry.startedAt + job.timeoutSeconds * 1000) {
        entry.status = 'cleanup'; entry.targetState = 'failed'; entry.retryable = false; entry.reason = 'job_deadline';
      } else {
        try {
          result = await call(adapter.inspect.bind(adapter), { job, runId: entry.runId, idempotencyKey: entry.idempotencyKey }, Math.min(30000, entry.startedAt + job.timeoutSeconds * 1000 - now));
          fail(result && ['running', 'completed', 'failed'].includes(result.status), 'Invalid provider inspection');
          if (result.costUsd !== undefined) fail(amount(result.costUsd) && result.costUsd <= job.maxCostUsd, 'Provider cost breach');
          if (result.status !== 'running') {
            entry.status = 'cleanup'; entry.targetState = result.status === 'completed' && verified(result.completion, job, entry) ? 'succeeded' : 'failed';
            entry.retryable = result.status === 'failed'; entry.reason = entry.targetState === 'succeeded' ? 'verified_completion' : result.status === 'failed' ? 'provider_failed' : 'unverified_completion';
            if (entry.targetState === 'succeeded') entry.completionSha256 = sha(JSON.stringify(result.completion));
          }
        } catch { entry.status = 'cleanup'; entry.targetState = 'failed'; entry.retryable = false; entry.reason = 'inspection_failed_or_invalid'; }
      }
    } else if (entry.status === 'cleanup') {
      await persist(config, state); // Cleanup intent is retained until resource absence is confirmed by the host.
      try {
        const result = await call(adapter.cleanup.bind(adapter), { job, runId: entry.runId }, 30000);
        if (result?.released === true) {
          entry.history ??= [];
          entry.history.push({ attempt: entry.attempts, runId: entry.runId, idempotencyKey: entry.idempotencyKey,
            outcome: entry.targetState, reason: entry.reason, cleanupConfirmed: true,
            ...(entry.completionSha256 ? { completionSha256: entry.completionSha256 } : {}) });
          entry.status = entry.targetState;
          if (entry.status === 'failed' && entry.retryable && entry.attempts < config.maxAttempts) entry.status = 'queued';
          entry.cleanupConfirmed = true;
        } else entry.reason = 'cleanup_unconfirmed';
      } catch { entry.reason = 'cleanup_unconfirmed'; }
    }
    if (entry.status === 'running') state.nextTickAt = Math.min(state.nextTickAt, entry.startedAt + job.timeoutSeconds * 1000);
    if (entry.status === 'cleanup' && entry.reason !== 'cleanup_unconfirmed') state.nextTickAt = now;
    await persist(config, state); return state;
  } finally { await rm(lock, { recursive: true }); }
}
