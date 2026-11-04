import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { createVastAdapter, onstartScript } from '../src/vast-host-adapter.mjs';

const job = { id: 'paired', secretRef: 'ruos.vast.primary', resource: { offerId: '123' }, maxCostUsd: 1.6, timeoutSeconds: 14400,
  model: { repo: 'Qwen/Qwen3.5-4B', revision: '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' } };

async function setup({ dph = 0.2, instances = [], guardCode = 0, remoteState = 'running', reportsMatch = true, extract, host = {} } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'vast-adapter-'));
  const config = { schema: 'ruvllm-vast-host/v1', apiBase: 'https://vast.test', image: 'pytorch/pytorch@sha256:' + 'a'.repeat(64),
    repoUrl: 'https://github.com/ruvnet/ruvllm.git', commit: 'b'.repeat(40), bundle: 'v2', diskGb: 40, armDeadlineSeconds: 5400,
    guardGraceSeconds: 900, resultsRoot: root, sshKey: '/k', repoRoot: '/repo', python: '/py', node: '/node',
    secrets: { 'ruos.vast.primary': { file: join(root, 'key') } }, sha256: 'c'.repeat(64), ...host };
  await writeFile(join(root, 'key'), 'k'.repeat(32), { mode: 0o600 });
  const calls = [], rows = [...instances];
  const fetchImpl = async (url, init) => {
    const path = new URL(url).pathname; calls.push([init.method, path]);
    const reply = body => ({ ok: true, status: 200, json: async () => body });
    assert.equal(init.headers.Authorization, 'Bearer ' + 'k'.repeat(32));
    if (path === '/api/v1/instances/') return reply({ instances: rows });
    if (path === '/api/v0/bundles/') return reply({ offers: [{ id: 123, dph_total: dph, gpu_name: 'RTX 3090' }] });
    if (init.method === 'PUT') { rows.push({ id: 77, label: JSON.parse(init.body).label, dph_total: dph }); return reply({ success: true, new_contract: 77 }); }
    if (init.method === 'DELETE') { rows.splice(0, rows.length); return reply({ success: true }); }
    throw new Error('unexpected ' + path);
  };
  const run = async (command, args, options) => {
    calls.push([command, args.at(-1)]);
    if (command === 'systemd-run') return { code: guardCode, stdout: '' };
    if (command === 'ssh' && args.at(-1).startsWith('cat ')) return { code: 0, stdout: JSON.stringify({ state: remoteState }) + '\n' };
    if (command === 'ssh') { await writeFile(options.stdoutFile, 'tar'); return { code: 0 }; }
    if (command === 'tar' && extract) { await extract(args[3]); return { code: 0 }; }
    if (command === 'tar') {
      const out = join(args[3], 'out', 'scored'); await mkdir(out, { recursive: true });
      await writeFile(join(out, 'report.json'), JSON.stringify({ evidence_mode: 'gpu-inference', reasons: reportsMatch ? [] : ['x'] }));
      return { code: 0 };
    }
    if (command === '/py') {
      const out = args[args.indexOf('--out') + 1]; await mkdir(out);
      await writeFile(join(out, 'report.json'), JSON.stringify({ evidence_mode: 'gpu-inference', reasons: [] }));
      return { code: 0 };
    }
    return { code: 0 };
  };
  const adapter = createVastAdapter({ fetchImpl, run, now: () => 1_800_000_000_000, readConfig: async () => config });
  return { adapter, calls, rows, root, config };
}

test('refuses an offer whose worst-case cost exceeds the job ceiling', async () => {
  const { adapter, calls } = await setup({ dph: 0.5 });
  await assert.rejects(adapter.submit({ job, idempotencyKey: 'key-1' }), { code: 'VAST_COST_CEILING' });
  assert.ok(!calls.some(([method]) => method === 'PUT'));
});

test('submits once, arms the guard and records a private receipt', async () => {
  const { adapter, calls, root } = await setup();
  assert.deepEqual(await adapter.submit({ job, idempotencyKey: 'key-1' }), { runId: '77' });
  assert.ok(calls.some(([command]) => command === 'systemd-run'));
  const receipt = JSON.parse(await readFile(join(root, 'key-1', 'submission.json'), 'utf8'));
  assert.equal(receipt.worstCaseUsd, 0.8);
  assert.equal(JSON.stringify(receipt).includes('k'.repeat(32)), false);
});

test('a retried submit finds the labeled instance instead of renting again', async () => {
  const { adapter, calls } = await setup({ instances: [{ id: 55, label: 'ruvllm-key-1' }] });
  assert.deepEqual(await adapter.submit({ job, idempotencyKey: 'key-1' }), { runId: '55' });
  assert.ok(!calls.some(([method]) => method === 'PUT'));
});

test('an unarmed guard destroys the new rental and fails', async () => {
  const { adapter, rows } = await setup({ guardCode: 1 });
  await assert.rejects(adapter.submit({ job, idempotencyKey: 'key-1' }), { code: 'VAST_GUARD_UNAVAILABLE' });
  assert.equal(rows.length, 0);
});

test('inspect reports running, then a verified completion only after local re-scoring matches', async () => {
  const running = await setup({ instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000 - 3600, actual_status: 'running', ssh_host: 'h', ssh_port: 1 }] });
  await mkdir(join(running.root, 'key-1'));
  assert.deepEqual(await running.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' }), { status: 'running', costUsd: 0.2 });
  for (const match of [true, false]) {
    const done = await setup({ remoteState: 'done', reportsMatch: match,
      instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000, actual_status: 'running', ssh_host: 'h', ssh_port: 1 }] });
    await mkdir(join(done.root, 'key-1'));
    const result = await done.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' });
    assert.equal(result.status, 'completed');
    assert.equal(result.completion.completionVerified, match);
    assert.equal(result.completion.idempotencyKey, 'key-1');
  }
});

test('inspect fails a run whose accrued cost passes its ceiling', async () => {
  const { adapter } = await setup({ instances: [{ id: 77, dph_total: 1, start_date: 1_800_000_000 - 7200, actual_status: 'running' }] });
  assert.equal((await adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
});

test('cleanup releases only after the instance is confirmed absent', async () => {
  const { adapter, root } = await setup({ instances: [{ id: 77 }] });
  assert.deepEqual(await adapter.cleanup({ job, runId: '77' }), { released: true });
  assert.equal(JSON.parse(await readFile(join(root, 'cleanup-77.json'), 'utf8')).confirmedAbsent, true);
});

test('onstart pins the commit and passes no credential', () => {
  const script = onstartScript({ repoUrl: 'https://github.com/ruvnet/ruvllm.git', commit: 'b'.repeat(40), armDeadlineSeconds: 5400, bundle: 'v2' });
  assert.match(script, /checkout --quiet b{40}/);
  assert.doesNotMatch(script, /api[_-]?key|Bearer/i);
});

const digest = text => createHash('sha256').update(text).digest('hex');

async function trainingEvidence(root, { payloadSha = 'p'.repeat(64), tamper = false } = {}) {
  const out = join(root, 'out', 'training-v1');
  const files = {};
  for (const step of ['step-032', 'step-064', 'step-096']) {
    await mkdir(join(out, step), { recursive: true });
    for (const name of ['adapter_model.safetensors', 'adapter_config.json']) {
      await writeFile(join(out, step, name), step + name);
      files[`${step}/${name}`] = digest(step + name);
    }
  }
  const manifest = JSON.stringify({ model: 'Qwen/Qwen3.5-4B', revision: job.model.revision, steps: 96, training_sha256: payloadSha, files });
  await writeFile(join(out, 'manifest.json'), manifest);
  if (tamper) await writeFile(join(out, 'step-064', 'adapter_model.safetensors'), 'tampered');
  await mkdir(join(root, 'run'), { recursive: true });
  await writeFile(join(root, 'run', 'train.log'), [
    'RUFLO_TRAIN ' + JSON.stringify({ kind: 'validation', data: { step: 64, loss: 0.1 } }),
    'RUFLO_TRAIN ' + JSON.stringify({ kind: 'completed', data: { steps: 96, manifest_sha256: digest(manifest) } }), ''].join('\n'));
}

test('a training job runs the pinned trainer script and verifies the manifest against the committed payload', async () => {
  const trainJob = { ...job, id: 'train', task: 'train' };
  for (const tamper of [false, true]) {
    const repo = await mkdtemp(join(tmpdir(), 'vast-repo-'));
    const payload = 'experiments/model-improvement/training/pool-v4';
    await mkdir(join(repo, payload), { recursive: true });
    await writeFile(join(repo, payload, 'bridge.json'), JSON.stringify({ training_sha256: 'p'.repeat(64) }));
    const { adapter, calls, root } = await setup({ remoteState: 'done', host: { repoRoot: repo, trainPayload: payload },
      extract: dir => trainingEvidence(dir, { tamper }),
      instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000, actual_status: 'running', ssh_host: 'h', ssh_port: 1 }] });
    await mkdir(join(root, 'key-1'));
    const result = await adapter.inspect({ job: trainJob, runId: '77', idempotencyKey: 'key-1' });
    assert.equal(result.status, 'completed');
    assert.equal(result.completion.completionVerified, !tamper);
    assert.ok(calls.some(([command, remote]) => command === 'ssh' && remote.includes('train.log')));
  }
  const script = onstartScript({ repoUrl: 'https://github.com/ruvnet/ruvllm.git', commit: 'b'.repeat(40), armDeadlineSeconds: 7200,
    bundle: 'v4', trainPayload: 'experiments/model-improvement/training/pool-v4' }, 'train');
  assert.match(script, /remote-train\.sh b{40} 7200 experiments\/model-improvement\/training\/pool-v4/);
  assert.throws(() => onstartScript({ commit: 'b'.repeat(40) }, 'train'), { code: 'VAST_HOST_CONFIG_PAYLOAD' });
  assert.match(onstartScript({ repoUrl: 'https://github.com/ruvnet/ruvllm.git', commit: 'b'.repeat(40), armDeadlineSeconds: 5400,
    bundle: 'v4', adapterDir: 'coding-4b-v4' }), /remote-run\.sh b{40} 5400 v4 coding-4b-v4/);
});

test('a container runtime error on the host fails the attempt instead of waiting for the deadline', async () => {
  const { adapter } = await setup({ instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000, actual_status: 'loading',
    status_msg: 'Error response from daemon: unknown or invalid runtime name: nvidia' }] });
  assert.equal((await adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
  const loading = await setup({ instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000, actual_status: 'loading', status_msg: 'Download complete' }] });
  assert.equal((await loading.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'running');
});
