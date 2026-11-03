import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm, realpath, chmod, symlink, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { loadHostAdapter } from '../src/gpu-host.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const cli = fileURLToPath(new URL('../src/gpu-loop-cli.mjs', import.meta.url));
const source = `export async function submit({job,idempotencyKey}) { return {runId:'contract-test-run'}; }
export async function inspect() { return {status:'running'}; }
export async function cleanup() { return {released:true}; }\n`;

async function setup(t, code = source) {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'ruvllm-host-test-')));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const modulePath = join(directory, 'host.mjs');
  await writeFile(modulePath, code, { mode: 0o600 });
  return { directory, modulePath, sha256: digest(code) };
}

test('host loader verifies owner-only code and returns structured adapter functions', async t => {
  const fixture = await setup(t);
  const host = await loadHostAdapter(fixture);
  assert.equal(Object.isFrozen(host), true);
  assert.deepEqual(await host.submit({ job: { id: 'job-1' }, idempotencyKey: 'stable-key' }), { runId: 'contract-test-run' });
  assert.deepEqual(await host.inspect({ runId: 'contract-test-run' }), { status: 'running' });
});

test('pin mismatches refuse code execution before import', async t => {
  const fixture = await setup(t, `globalThis.__gpuHostUnexpectedExecution = true;\n${source}`);
  await assert.rejects(loadHostAdapter({ ...fixture, sha256: '0'.repeat(64) }), { code: 'HOST_ADAPTER_PIN_MISMATCH' });
  assert.equal(globalThis.__gpuHostUnexpectedExecution, undefined);
});

test('host loader rejects symlinks, non-private modules and invalid contracts', async t => {
  const fixture = await setup(t);
  const linked = join(fixture.directory, 'linked.mjs'); await symlink(fixture.modulePath, linked);
  await assert.rejects(loadHostAdapter({ ...fixture, modulePath: linked }), { code: 'HOST_ADAPTER_INVALID_PATH' });
  await chmod(fixture.modulePath, 0o644);
  await assert.rejects(loadHostAdapter(fixture), { code: 'HOST_ADAPTER_UNSAFE_FILE' });
  const incomplete = await setup(t, 'export async function submit() {}');
  await assert.rejects(loadHostAdapter(incomplete), { code: 'HOST_ADAPTER_INVALID_CONTRACT' });
});

test('host import and operation exceptions never expose raw provider details', async t => {
  const failedImport = await setup(t, 'throw new Error("fake-sensitive-provider-detail");');
  await assert.rejects(loadHostAdapter(failedImport), error => error.message === 'HOST_ADAPTER_UNAVAILABLE');
  const failedCall = await setup(t, source.replace("return {runId:'contract-test-run'}", 'throw new Error("fake-sensitive-provider-detail")'));
  const adapter = await loadHostAdapter(failedCall);
  await assert.rejects(adapter.submit({}), error => error.message === 'HOST_ADAPTER_OPERATION_FAILED');
});

test('CLI documents bounded loop and rejects invalid arguments without provider execution', async () => {
  const help = spawnSync(process.execPath, [cli, '--help'], { encoding: 'utf8' });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /max-ticks/);
  const invalid = spawnSync(process.execPath, [cli, 'run', '--max-ticks', 'Infinity'], { encoding: 'utf8' });
  assert.notEqual(invalid.status, 0);
  assert.doesNotMatch(invalid.stderr, /fake-sensitive-provider-detail/);
});

test('CLI tick and status use durable local state with a mock host only', async t => {
  const fixture = await setup(t);
  const configPath = join(fixture.directory, 'loop.json');
  const config = {
    schema: 'ruvllm-gpu-loop/v1', stateRoot: join(fixture.directory, 'state'), intervalSeconds: 60,
    maxAttempts: 1, budgetUsd: 1,
    jobs: [{ id: 'contract-only-job', provider: 'vast.ai', model: { repo: 'Qwen/Qwen3.5-4B', revision: 'a'.repeat(40) },
      task: 'evaluate', secretRef: 'vast-api-key', resource: { instanceId: '1234' }, maxCostUsd: 1, timeoutSeconds: 60 }],
  };
  await mkdir(config.stateRoot, { mode: 0o700 });
  await writeFile(configPath, JSON.stringify(config), { mode: 0o600 });
  const tick = spawnSync(process.execPath, [cli, 'tick', '--config', configPath,
    '--adapter', fixture.modulePath, '--adapter-sha256', fixture.sha256], { encoding: 'utf8', timeout: 10000 });
  assert.equal(tick.status, 0, tick.stderr);
  const first = JSON.parse(tick.stdout);
  assert.equal(first.jobs[0].runId, 'contract-test-run');
  assert.equal(first.productionActivation, false);
  const status = spawnSync(process.execPath, [cli, 'status', '--config', configPath], { encoding: 'utf8', timeout: 10000 });
  assert.equal(status.status, 0, status.stderr);
  assert.deepEqual(JSON.parse(status.stdout), first);
  assert.doesNotMatch(await readFile(configPath, 'utf8'), /api_key_value/);
});

test('bounded run mode advances scheduled ticks without submitting the same running job twice', async t => {
  const fixture = await setup(t);
  const configPath = join(fixture.directory, 'loop.json');
  const stateRoot = join(fixture.directory, 'state');
  await mkdir(stateRoot, { mode: 0o700 });
  await writeFile(configPath, JSON.stringify({ schema: 'ruvllm-gpu-loop/v1', stateRoot, intervalSeconds: 1,
    maxAttempts: 1, budgetUsd: 1,
    jobs: [{ id: 'bounded-loop-job', provider: 'vast.ai', model: { repo: 'Qwen/Qwen3.5-4B', revision: 'b'.repeat(40) },
      task: 'evaluate', secretRef: 'vast-reference-only', resource: { instanceId: '1234' }, maxCostUsd: 1, timeoutSeconds: 60 }],
  }), { mode: 0o600 });
  const args = [cli, 'run', '--config', configPath, '--adapter', fixture.modulePath, '--adapter-sha256', fixture.sha256];
  const invalid = spawnSync(process.execPath, [...args, '--max-ticks', 'Infinity'], { encoding: 'utf8', timeout: 10000 });
  assert.notEqual(invalid.status, 0);
  const run = spawnSync(process.execPath, [...args, '--max-ticks', '2'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(run.status, 0, run.stderr);
  const ticks = run.stdout.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(ticks.length, 2);
  assert.equal(ticks[0].jobs[0].runId, ticks[1].jobs[0].runId);
  assert.equal(ticks[1].jobs[0].attempts, 1);
  assert.equal(ticks[1].reservedUsd, 1);
});

test('run mode wakes for a job deadline before a longer schedule interval', async t => {
  const fixture = await setup(t);
  const configPath = join(fixture.directory, 'loop.json');
  const stateRoot = join(fixture.directory, 'state');
  await mkdir(stateRoot, { mode: 0o700 });
  await writeFile(configPath, JSON.stringify({ schema: 'ruvllm-gpu-loop/v1', stateRoot, intervalSeconds: 60,
    maxAttempts: 1, budgetUsd: 1,
    jobs: [{ id: 'deadline-loop-job', provider: 'vast.ai', model: { repo: 'Qwen/Qwen3.5-4B', revision: 'c'.repeat(40) },
      task: 'evaluate', secretRef: 'vast-reference-only', resource: { instanceId: '1234' }, maxCostUsd: 1, timeoutSeconds: 1 }],
  }), { mode: 0o600 });
  const run = spawnSync(process.execPath, [cli, 'run', '--config', configPath,
    '--adapter', fixture.modulePath, '--adapter-sha256', fixture.sha256, '--max-ticks', '2'],
  { encoding: 'utf8', timeout: 10000 });
  assert.equal(run.status, 0, run.stderr);
  const ticks = run.stdout.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(ticks.length, 2);
  assert.equal(ticks[1].jobs[0].status, 'cleanup');
  assert.equal(ticks[1].jobs[0].reason, 'job_deadline');
});
