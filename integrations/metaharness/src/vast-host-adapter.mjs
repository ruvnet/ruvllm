// Vast.ai host adapter for the scheduled GPU loop (ADR-015, ADR-017). Trusted
// operator code: it resolves the job's secretRef from an owner-only host config,
// rents exactly the configured offer, never reuses a shared instance, and only
// reports a verified completion after re-scoring the retrieved raw outputs here.
// No effect happens at import; `node vast-host-adapter.mjs guard ID REF` is the
// independent destroy timer armed for every rental.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { constants, createWriteStream } from 'node:fs';
import { mkdir, open, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { isDeepStrictEqual } from 'node:util';
import { isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HOST_CONFIG_ENV = 'RUVLLM_VAST_HOST_CONFIG';
const SELF = fileURLToPath(import.meta.url);
const INSPECT_BUDGET_MS = 24000;
const fail = code => { const error = new Error(code); error.code = code; throw error; };
const need = (ok, code) => { if (!ok) fail(code); };
const abs = value => typeof value === 'string' && isAbsolute(value) && resolve(value) === value;
const ceilMicro = usd => Math.ceil(usd * 1e6) / 1e6;

async function privateJson(path) {
  need(abs(path), 'VAST_HOST_CONFIG_PATH');
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const info = await handle.stat();
    need(info.isFile() && (info.mode & 0o077) === 0 && info.uid === process.getuid() && info.size <= 65536, 'VAST_HOST_CONFIG_UNSAFE');
    const bytes = await handle.readFile();
    return { value: JSON.parse(bytes), sha256: createHash('sha256').update(bytes).digest('hex') };
  } finally { await handle.close(); }
}

export async function readHostConfig(path = process.env[HOST_CONFIG_ENV]) {
  const { value: config, sha256 } = await privateJson(path);
  need(config?.schema === 'ruvllm-vast-host/v1', 'VAST_HOST_CONFIG_SCHEMA');
  need(/^https:\/\/[a-z0-9.-]+$/.test(config.apiBase), 'VAST_HOST_CONFIG_API');
  need(/^[a-z0-9./_-]+@sha256:[a-f0-9]{64}$/.test(config.image), 'VAST_HOST_CONFIG_IMAGE');
  need(/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$/.test(config.repoUrl) && /^[a-f0-9]{40}$/.test(config.commit), 'VAST_HOST_CONFIG_SOURCE');
  need(/^v[0-9]+$/.test(config.bundle), 'VAST_HOST_CONFIG_BUNDLE');
  need(Number.isInteger(config.diskGb) && config.diskGb >= 20 && config.diskGb <= 200, 'VAST_HOST_CONFIG_DISK');
  need(Number.isInteger(config.armDeadlineSeconds) && config.armDeadlineSeconds >= 60 && config.armDeadlineSeconds <= 43200, 'VAST_HOST_CONFIG_DEADLINE');
  need(Number.isInteger(config.guardGraceSeconds) && config.guardGraceSeconds >= 60 && config.guardGraceSeconds <= 3600, 'VAST_HOST_CONFIG_GUARD');
  need(['resultsRoot', 'sshKey', 'repoRoot', 'python', 'node'].every(key => abs(config[key])), 'VAST_HOST_CONFIG_PATHS');
  need(config.adapterDir === undefined || (/^[a-z0-9][a-z0-9._-]{0,63}$/.test(config.adapterDir) && !config.adapterDir.includes('..')), 'VAST_HOST_CONFIG_ADAPTER');
  need(config.trainPayload === undefined || /^experiments\/model-improvement\/training\/[a-z0-9][a-z0-9._-]{0,63}$/.test(config.trainPayload), 'VAST_HOST_CONFIG_PAYLOAD');
  need(config.secrets && typeof config.secrets === 'object' && Object.values(config.secrets).every(entry => abs(entry?.file)), 'VAST_HOST_CONFIG_SECRETS');
  return Object.freeze({ ...config, sha256 });
}

async function apiKey(config, ref) {
  const entry = Object.hasOwn(config.secrets, ref) ? config.secrets[ref] : null;
  need(entry, 'VAST_SECRET_REF_UNKNOWN');
  const { mode, uid } = await stat(entry.file);
  need((mode & 0o077) === 0 && uid === process.getuid(), 'VAST_SECRET_FILE_UNSAFE');
  const key = (await readFile(entry.file, 'utf8')).trim();
  need(/^[A-Za-z0-9_-]{16,256}$/.test(key), 'VAST_SECRET_INVALID');
  return key;
}

// Bounded child process without a shell. Output is collected, never echoed.
function defaultRun(command, args, { timeoutMs, cwd, stdoutFile, input } = {}) {
  return new Promise(resolvePromise => {
    const child = spawn(command, args, { cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    const chunks = [], errors = [];
    let size = 0;
    const sink = stdoutFile ? createWriteStream(stdoutFile, { flags: 'wx', mode: 0o600 }) : null;
    const flushed = sink ? new Promise(done => { sink.on('close', done); sink.on('error', done); }) : Promise.resolve();
    const timer = setTimeout(() => child.kill('SIGKILL'), Math.max(1, timeoutMs ?? 10000));
    if (sink) child.stdout.pipe(sink);
    else child.stdout.on('data', chunk => { size += chunk.length; if (size <= 8 << 20) chunks.push(chunk); });
    child.stderr.on('data', chunk => { if (errors.length < 64) errors.push(chunk); });
    child.on('error', () => {});
    child.on('close', async code => {
      clearTimeout(timer);
      await flushed;
      resolvePromise({ code, stdout: Buffer.concat(chunks).toString('utf8'), stderr: Buffer.concat(errors).toString('utf8') });
    });
    child.stdin.end(input ?? '');
  });
}

async function writePrivate(path, value) {
  await writeFile(path, typeof value === 'string' ? value : JSON.stringify(value, null, 2) + '\n', { mode: 0o600 });
}

export function onstartScript(config, task = 'evaluate') {
  const scripts = '/workspace/ruvllm/experiments/model-improvement/vast';
  if (task === 'train') need(config.trainPayload, 'VAST_HOST_CONFIG_PAYLOAD');
  const run = task === 'train'
    ? `${scripts}/remote-train.sh ${config.commit} ${config.armDeadlineSeconds} ${config.trainPayload}`
    : `${scripts}/remote-run.sh ${config.commit} ${config.armDeadlineSeconds} ${config.bundle} ${config.adapterDir ?? 'coding-4b'}`;
  const failed = `grep -q '"failed"' /workspace/run/status.json 2>/dev/null || printf '{"schema":"ruvllm-vast-remote-status/v1","state":"failed","step":"bootstrap"}\\n' > /workspace/run/status.json`;
  return ['#!/bin/bash', 'mkdir -p /workspace/run',
    `nohup bash -c '(export DEBIAN_FRONTEND=noninteractive; (command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)) && git clone --quiet ${config.repoUrl} /workspace/ruvllm && git -C /workspace/ruvllm checkout --quiet ${config.commit} && bash ${run}) >/workspace/run/remote.log 2>&1 || { ${failed.replaceAll("'", "'\\''")}; }' >/dev/null 2>&1 &`, ''].join('\n');
}

export function createVastAdapter({ fetchImpl = globalThis.fetch, run = defaultRun, now = () => Date.now(), readConfig = readHostConfig } = {}) {
  async function api(config, key, method, path, { body, query, signal } = {}) {
    const url = new URL(config.apiBase + path);
    for (const [name, value] of Object.entries(query ?? {})) url.searchParams.set(name, JSON.stringify(value));
    const response = await fetchImpl(url, { method, signal, headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    if (response.status === 404) return null;
    need(response.ok, `VAST_HTTP_${response.status}`);
    return response.json();
  }
  async function instances(config, key, signal) {
    const rows = [];
    let after;
    for (let page = 0; page < 40; page++) {
      const data = await api(config, key, 'GET', '/api/v1/instances/', { signal,
        query: { select_filters: {}, order_by: [{ col: 'id', dir: 'asc' }], limit: 25, ...(after ? { after_token: after } : {}) } });
      rows.push(...(data?.instances ?? []));
      if (!data?.next_token) return rows;
      after = data.next_token;
    }
    fail('VAST_LIST_UNBOUNDED');
  }
  const attemptDir = (config, key) => join(config.resultsRoot, key);
  const ssh = (config, instance, remote) => ['-i', config.sshKey, '-p', String(instance.ssh_port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
    '-o', 'StrictHostKeyChecking=accept-new', '-o', `UserKnownHostsFile=${join(config.resultsRoot, `known_hosts-${instance.id}`)}`, '-o', 'IdentitiesOnly=yes',
    `root@${instance.ssh_host}`, remote];

  async function submit({ job, idempotencyKey, signal }) {
    const config = await readConfig(), key = await apiKey(config, job.secretRef);
    need(typeof job.resource.offerId === 'string' && /^[0-9]{1,12}$/.test(job.resource.offerId), 'VAST_OFFER_REQUIRED');
    const label = `ruvllm-${idempotencyKey}`;
    const existing = (await instances(config, key, signal)).find(row => row.label === label);
    if (existing) return { runId: String(existing.id) };
    // The bundles search ignores `id`; `ask_contract_id` is the offer identity filter.
    const offers = (await api(config, key, 'POST', '/api/v0/bundles/', { signal, body: { ask_contract_id: { eq: Number(job.resource.offerId) },
      rentable: { eq: true }, type: 'on-demand', allocated_storage: config.diskGb, limit: 2 } }))?.offers ?? [];
    need(offers.length === 1 && String(offers[0].id) === job.resource.offerId
      && Number.isFinite(offers[0].dph_total) && offers[0].dph_total > 0, 'VAST_OFFER_UNAVAILABLE');
    const offer = offers[0];
    // Spending ceiling: the worst case (full timeout at the offer's total hourly
    // price, storage included) must fit the job's reserved ceiling before renting.
    const worstCaseUsd = ceilMicro(offer.dph_total * job.timeoutSeconds / 3600);
    need(worstCaseUsd <= job.maxCostUsd, 'VAST_COST_CEILING');
    await mkdir(attemptDir(config, idempotencyKey), { mode: 0o700 });
    const created = await api(config, key, 'PUT', `/api/v0/asks/${offer.id}/`, { signal, body: { client_id: 'me', image: config.image,
      env: {}, price: null, disk: config.diskGb, label, onstart: onstartScript(config, job.task), runtype: 'ssh', force: false, cancel_unavail: true } });
    need(created?.success === true && Number.isSafeInteger(created.new_contract), 'VAST_CREATE_FAILED');
    const instanceId = created.new_contract;
    // Independent of this scheduler: a user timer destroys the rental after
    // timeout + grace even if no further tick ever runs.
    const guard = await run('systemd-run', ['--user', '--quiet', `--unit=ruvllm-vast-guard-${instanceId}`,
      `--on-active=${job.timeoutSeconds + config.guardGraceSeconds}s`, `--setenv=${HOST_CONFIG_ENV}=${process.env[HOST_CONFIG_ENV] ?? ''}`,
      config.node, SELF, 'guard', String(instanceId), job.secretRef], { timeoutMs: 8000 });
    await writePrivate(join(attemptDir(config, idempotencyKey), 'submission.json'), { schema: 'ruvllm-vast-submission/v1',
      jobId: job.id, idempotencyKey, label, instanceId, offerId: offer.id, gpuName: offer.gpu_name, dphTotal: offer.dph_total,
      worstCaseUsd, maxCostUsd: job.maxCostUsd, timeoutSeconds: job.timeoutSeconds, image: config.image, commit: config.commit,
      task: job.task, bundle: config.bundle, adapterDir: config.adapterDir ?? 'coding-4b', trainPayload: config.trainPayload ?? null, hostConfigSha256: config.sha256, guardArmed: guard.code === 0, submittedAt: new Date(now()).toISOString() });
    if (guard.code !== 0) {
      await api(config, key, 'DELETE', `/api/v0/instances/${instanceId}/`, { body: {} });
      fail('VAST_GUARD_UNAVAILABLE');
    }
    return { runId: String(instanceId) };
  }

  async function retrieve(config, instance, dir, started) {
    const left = () => INSPECT_BUDGET_MS - (now() - started);
    const archive = join(dir, 'evidence.tar');
    await rm(archive, { force: true }); await rm(join(dir, 'evidence'), { recursive: true, force: true });
    if (left() < 4000) return false;
    const pulled = await run('ssh', ssh(config, instance, 'tar -C /workspace -cf - out run'), { timeoutMs: left() - 3000, stdoutFile: archive });
    if (pulled.code !== 0) return false;
    await mkdir(join(dir, 'evidence'), { mode: 0o700 });
    return (await run('tar', ['-xf', archive, '-C', join(dir, 'evidence'), '--no-same-owner', '--no-same-permissions'], { timeoutMs: 5000 })).code === 0;
  }

  async function verify(config, job, dir) {
    const evidence = join(dir, 'evidence', 'out'), scored = join(dir, 'local-scored');
    await rm(scored, { recursive: true, force: true });
    const bundle = join(config.repoRoot, 'experiments/model-improvement/frozen', config.bundle);
    const result = await run(config.python, [join(config.repoRoot, 'experiments/model-improvement/measurement.py'), '--bundle', bundle,
      '--base-run', join(evidence, 'base'), '--adapter-run', join(evidence, 'adapter'), '--out', scored], { timeoutMs: 15000, cwd: config.repoRoot });
    if (result.code !== 0) return { ok: false, reason: 'local_rescore_failed' };
    const local = JSON.parse(await readFile(join(scored, 'report.json'), 'utf8'));
    const remote = JSON.parse(await readFile(join(evidence, 'scored', 'report.json'), 'utf8'));
    const ok = isDeepStrictEqual(local, remote) && local.evidence_mode === 'gpu-inference';
    await writePrivate(join(dir, 'verification.json'), { schema: 'ruvllm-vast-verification/v1', jobId: job.id,
      localRescoreMatchesRemote: isDeepStrictEqual(local, remote), evidenceMode: local.evidence_mode,
      researchImprovementConfirmed: local.research_improvement_confirmed, reasons: local.reasons });
    return { ok, reason: ok ? 'verified' : 'report_mismatch' };
  }

  // Training evidence: the trainer's own manifest must hash every retrieved file,
  // bind the committed payload, and match the completed event in the event log.
  async function verifyTraining(config, job, dir) {
    const out = join(dir, 'evidence', 'out', 'training-v1');
    const sha = async path => createHash('sha256').update(await readFile(path)).digest('hex');
    const checks = {};
    try {
      const bridge = JSON.parse(await readFile(join(config.repoRoot, config.trainPayload, 'bridge.json'), 'utf8'));
      const manifestBytes = await readFile(join(out, 'manifest.json'));
      const manifest = JSON.parse(manifestBytes);
      const events = (await readFile(join(dir, 'evidence', 'run', 'train.log'), 'utf8')).split('\n')
        .filter(line => line.startsWith('RUFLO_TRAIN ')).map(line => JSON.parse(line.slice(12)));
      const completed = events.find(event => event.kind === 'completed');
      checks.payloadBound = manifest.training_sha256 === bridge.training_sha256;
      checks.manifestEvent = completed?.data?.manifest_sha256 === createHash('sha256').update(manifestBytes).digest('hex');
      checks.steps = manifest.steps === 96 && manifest.revision === job.model.revision && manifest.model === job.model.repo;
      const files = Object.entries(manifest.files ?? {});
      checks.files = files.length >= 6 && (await Promise.all(files.map(async ([name, digest]) =>
        !name.split('/').includes('..') && await sha(join(out, name)) === digest))).every(Boolean);
      checks.validation = events.filter(event => event.kind === 'validation').map(event => event.data);
    } catch (error) { checks.error = error.code ?? error.name; }
    const ok = Boolean(checks.payloadBound && checks.manifestEvent && checks.steps && checks.files && !checks.error);
    await writePrivate(join(dir, 'verification.json'), { schema: 'ruvllm-vast-training-verification/v1', jobId: job.id, ...checks, ok });
    return { ok };
  }

  async function inspect({ job, runId, idempotencyKey, signal }) {
    const started = now(), config = await readConfig(), key = await apiKey(config, job.secretRef);
    const instance = (await instances(config, key, signal)).find(row => String(row.id) === runId);
    if (!instance) return { status: 'failed' };
    const costUsd = ceilMicro(instance.dph_total * Math.max(0, now() / 1000 - (instance.start_date ?? now() / 1000)) / 3600);
    if (costUsd > job.maxCostUsd) return { status: 'failed', costUsd };
    // A host whose container runtime cannot start the image never recovers; fail now instead of at the deadline.
    if (instance.actual_status !== 'running' && /error response from daemon|oci runtime|invalid runtime/i.test(instance.status_msg ?? '')) return { status: 'failed', costUsd };
    if (instance.actual_status !== 'running' || !instance.ssh_host || !instance.ssh_port) return { status: 'running', costUsd };
    const dir = attemptDir(config, idempotencyKey);
    const probe = await run('ssh', ssh(config, instance, `cat /workspace/run/status.json && echo && { tail -c 400 /workspace/run/${job.task === 'train' ? 'train' : 'run-pair'}.log 2>/dev/null || true; }`), { timeoutMs: 10000 });
    if (probe.code !== 0) return { status: 'running', costUsd };
    let remote;
    try { remote = JSON.parse(probe.stdout.split('\n')[0]); } catch { return { status: 'running', costUsd }; }
    await writePrivate(join(dir, 'progress.json'), { at: new Date(now()).toISOString(), costUsd, remote, tail: probe.stdout.split('\n').slice(1).join('\n').slice(-400) });
    if (remote.state === 'running') return { status: 'running', costUsd };
    if (remote.state === 'failed') { await retrieve(config, instance, dir, started); return { status: 'failed', costUsd }; }
    if (remote.state !== 'done') return { status: 'running', costUsd };
    // Retain raw outputs locally before any terminal report; retry next tick on a slow pull.
    if (!await retrieve(config, instance, dir, started)) return { status: 'running', costUsd };
    const checked = job.task === 'train' ? await verifyTraining(config, job, dir) : await verify(config, job, dir);
    return { status: 'completed', costUsd, completion: { completionVersion: 1, completionVerified: checked.ok, status: checked.ok ? 'ok' : 'error',
      exitCode: checked.ok ? 0 : 1, assertionsVerified: checked.ok, evidenceComplete: checked.ok, truncated: false, jobId: job.id,
      modelRepo: job.model.repo, modelRevision: job.model.revision, runId, idempotencyKey } };
  }

  async function destroyConfirmed(config, key, instanceId, signal) {
    await api(config, key, 'DELETE', `/api/v0/instances/${instanceId}/`, { body: {}, signal });
    for (let check = 0; check < 6; check++) {
      if (!(await instances(config, key, signal)).some(row => String(row.id) === String(instanceId))) return true;
      await new Promise(done => setTimeout(done, 2500));
    }
    return false;
  }

  async function cleanup({ job, runId, signal }) {
    const config = await readConfig(), key = await apiKey(config, job.secretRef);
    const released = await destroyConfirmed(config, key, runId, signal);
    if (released) await run('systemctl', ['--user', 'stop', `ruvllm-vast-guard-${runId}.timer`], { timeoutMs: 5000 });
    await writePrivate(join(config.resultsRoot, `cleanup-${runId}.json`), { schema: 'ruvllm-vast-cleanup/v1', jobId: job.id,
      instanceId: runId, confirmedAbsent: released, at: new Date(now()).toISOString() });
    return { released };
  }

  return { submit, inspect, cleanup, destroyConfirmed, instances };
}

const defaults = createVastAdapter();
export const submit = input => defaults.submit(input);
export const inspect = input => defaults.inspect(input);
export const cleanup = input => defaults.cleanup(input);

if (process.argv[1] === SELF && process.argv[2] === 'guard') {
  const [instanceId, ref] = process.argv.slice(3);
  const config = await readHostConfig();
  const released = await defaults.destroyConfirmed(config, await apiKey(config, ref), instanceId);
  await writePrivate(join(config.resultsRoot, `guard-${instanceId}.json`), { schema: 'ruvllm-vast-guard/v1', instanceId,
    confirmedAbsent: released, at: new Date().toISOString() });
  process.exitCode = released ? 0 : 1;
}
