import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, chmod } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { createVastAdapter, onstartScript, readHostConfig } from '../src/vast-host-adapter.mjs';
import { tarArchive } from './tar-fixture.mjs';

const job = { id: 'paired', secretRef: 'ruos.vast.primary', resource: { offerId: '123' }, maxCostUsd: 1.6, timeoutSeconds: 14400,
  model: { repo: 'Qwen/Qwen3.5-4B', revision: '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' } };

const EVIDENCE = tarArchive([{ name: 'out/', type: '5' }, { name: 'out/scored/report.json', body: '{}' }, { name: 'run/status.json', body: '{}' }]);

async function setup({ dph = 0.2, instances = [], guardCode = 0, remoteState = 'running', reportsMatch = true, extract, host = {}, archive = EVIDENCE, overflow = false } = {}) {
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
    if (command === 'ssh') { calls.push(['ssh-args', args]); await writeFile(options.stdoutFile, archive); return overflow ? { code: null, overflow: true } : { code: 0 }; }
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

const done = { remoteState: 'done', instances: [{ id: 77, dph_total: 0.2, start_date: 1_800_000_000, actual_status: 'running', ssh_host: 'h', ssh_port: 1 }] };

test('unsafe or oversized evidence archives are rejected before extraction and fail the job', async () => {
  const cases = {
    VAST_EVIDENCE_UNSAFE_PATH: [
      [{ name: 'out/../../etc/cron.d/x', body: 'x' }],
      [{ name: '/etc/passwd', body: 'x' }],
      [{ name: 'home/elsewhere', body: 'x' }],
      [{ name: 'out/' + 'a/'.repeat(80) + '../../../../x', body: 'x', longName: true }],
    ],
    VAST_EVIDENCE_ENTRY_TYPE: [
      [{ name: 'out/link', type: '2', linkname: '/etc/passwd' }],
      [{ name: 'out/hard', type: '1', linkname: 'out/x' }],
      [{ name: 'out/dev', type: '3' }],
    ],
  };
  for (const [reason, archives] of Object.entries(cases)) {
    for (const entries of archives) {
      const fixture = await setup({ ...done, archive: tarArchive(entries) });
      await mkdir(join(fixture.root, 'key-1'));
      assert.equal((await fixture.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed', JSON.stringify(entries));
      assert.ok(!fixture.calls.some(([command]) => command === 'tar'), 'never extracted');
      assert.equal(JSON.parse(await readFile(join(fixture.root, 'key-1', 'evidence-rejected.json'), 'utf8')).reason, reason);
    }
  }
  const garbage = await setup({ ...done, archive: Buffer.from('not a tar archive') });
  await mkdir(join(garbage.root, 'key-1'));
  assert.equal((await garbage.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
});

test('evidence size and entry caps apply to the pull and to the archive', async () => {
  const many = await setup({ ...done, host: { maxEvidenceEntries: 2 } });
  await mkdir(join(many.root, 'key-1'));
  assert.equal((await many.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
  assert.equal(JSON.parse(await readFile(join(many.root, 'key-1', 'evidence-rejected.json'), 'utf8')).reason, 'VAST_EVIDENCE_TOO_MANY_ENTRIES');
  const big = await setup({ ...done, host: { maxEvidenceBytes: 2048 } });
  await mkdir(join(big.root, 'key-1'));
  assert.equal((await big.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
  const pulled = await setup({ ...done, overflow: true });
  await mkdir(join(pulled.root, 'key-1'));
  assert.equal((await pulled.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' })).status, 'failed');
  assert.equal(JSON.parse(await readFile(join(pulled.root, 'key-1', 'evidence-rejected.json'), 'utf8')).reason, 'VAST_EVIDENCE_TOO_LARGE');
  const ssh = pulled.calls.find(([command]) => command === 'ssh-args');
  assert.ok(ssh, 'pull ran');
});

test('ssh host keys are pinned by default and trust on first use is opt-in', async () => {
  const pinned = await setup({ ...done, host: { sshKnownHosts: '/etc/ruvllm/known_hosts' } });
  await mkdir(join(pinned.root, 'key-1'));
  await pinned.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' });
  const pinnedArgs = pinned.calls.find(([command]) => command === 'ssh-args')[1];
  assert.ok(pinnedArgs.includes('StrictHostKeyChecking=yes') && pinnedArgs.includes('UserKnownHostsFile=/etc/ruvllm/known_hosts'));
  assert.ok(!pinnedArgs.some(arg => arg.includes('accept-new')));
  const tofu = await setup({ ...done, host: { sshTrustOnFirstUse: true } });
  await mkdir(join(tofu.root, 'key-1'));
  await tofu.adapter.inspect({ job, runId: '77', idempotencyKey: 'key-1' });
  assert.ok(tofu.calls.find(([command]) => command === 'ssh-args')[1].includes('StrictHostKeyChecking=accept-new'));
});

test('host config must pin ssh host keys or explicitly opt in to trust on first use', async () => {
  const root = await mkdtemp(join(tmpdir(), 'vast-config-'));
  const base = { schema: 'ruvllm-vast-host/v1', apiBase: 'https://vast.test', image: 'pytorch/pytorch@sha256:' + 'a'.repeat(64),
    repoUrl: 'https://github.com/ruvnet/ruvllm.git', commit: 'b'.repeat(40), bundle: 'v2', diskGb: 40, armDeadlineSeconds: 5400,
    guardGraceSeconds: 900, resultsRoot: root, sshKey: '/k', repoRoot: '/repo', python: '/py', node: '/node', secrets: {} };
  const write = async (name, value, mode = 0o600) => { const path = join(root, name); await writeFile(path, typeof value === 'string' ? value : JSON.stringify(value), { mode }); return path; };
  await assert.rejects(readHostConfig(await write('none.json', base)), { code: 'VAST_HOST_CONFIG_SSH_TRUST' });
  await assert.rejects(readHostConfig(await write('bad.json', { ...base, sshTrustOnFirstUse: 'yes' })), { code: 'VAST_HOST_CONFIG_SSH_TRUST' });
  assert.equal((await readHostConfig(await write('tofu.json', { ...base, sshTrustOnFirstUse: true }))).sshTrustOnFirstUse, true);
  const knownHosts = await write('known_hosts', '[ssh4.vast.ai]:12345 ssh-ed25519 AAAA\n', 0o600);
  assert.equal((await readHostConfig(await write('pinned.json', { ...base, sshKnownHosts: knownHosts }))).sshKnownHosts, knownHosts);
  const shared = await write('known_hosts_shared', '[h]:1 ssh-ed25519 AAAA\n', 0o666);
  await chmod(shared, 0o666);
  await assert.rejects(readHostConfig(await write('shared.json', { ...base, sshKnownHosts: shared })), { code: 'VAST_SSH_KNOWN_HOSTS_UNSAFE' });
  await assert.rejects(readHostConfig(await write('cap.json', { ...base, sshTrustOnFirstUse: true, maxEvidenceBytes: 10 })), { code: 'VAST_HOST_CONFIG_EVIDENCE' });
});
