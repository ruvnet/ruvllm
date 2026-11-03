import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createFixture } from '../../../integrations/metaharness/src/fixture.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const server = join(repo, 'plugins/ruflo-metaharness/scripts/server.mjs');

test('MCP parser rejects duplicate keys, malformed JSON and oversized requests', () => {
  const duplicate = spawnSync(process.execPath, [server], { input: '{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call"}\n', encoding: 'utf8' });
  assert.equal(JSON.parse(duplicate.stdout).error.code, -32700);
  const malformed = spawnSync(process.execPath, [server], { input: '{broken}\n', encoding: 'utf8' });
  assert.equal(JSON.parse(malformed.stdout).error.code, -32700);
  const oversized = spawnSync(process.execPath, [server], { input: 'x'.repeat(65537), encoding: 'utf8' });
  assert.equal(oversized.status, 1);
  assert.match(oversized.stderr, /request bound exceeded/);
});
function wire(configPath) {
  const child = spawn(process.execPath, [server], { env: { PATH: process.env.PATH, ...(configPath ? { RUVLLM_EXPERIMENT_CONFIG: configPath } : {}) }, stdio: ['pipe', 'pipe', 'pipe'] });
  let sequence = 0; let buffer = ''; let stderr = ''; const pending = new Map();
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdout.on('data', chunk => {
    buffer += chunk;
    while (buffer.includes('\n')) {
      const newline = buffer.indexOf('\n'); const value = JSON.parse(buffer.slice(0, newline)); buffer = buffer.slice(newline + 1);
      const waiter = pending.get(value.id);
      if (waiter) { pending.delete(value.id); clearTimeout(waiter.timer); waiter.resolve(value); }
    }
  });
  child.on('exit', () => { for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(new Error(`MCP exited: ${stderr}`)); } pending.clear(); });
  function rpc(method, params, additional = {}) {
    return new Promise((resolveResult, reject) => {
      const id = ++sequence;
      const timer = setTimeout(() => { pending.delete(id); child.kill(); reject(new Error(`MCP timeout: ${method}: ${stderr}`)); }, 30000);
      pending.set(id, { resolve: resolveResult, reject, timer });
      child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params, ...additional }) + '\n');
    });
  }
  return { rpc, async close() { child.stdin.end(); if (child.exitCode === null) await new Promise(done => child.once('exit', done)); assert.equal(stderr, ''); } };
}

test('MCP discovery works; unconfigured, arbitrary-command and activation requests refuse', async t => {
  const client = wire(); t.after(() => client.close());
  const init = await client.rpc('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '1' } });
  assert.equal(init.result.serverInfo.name, 'ruflo-metaharness');
  const listed = await client.rpc('tools/list', {});
  assert.deepEqual(listed.result.tools.map(v => v.name), ['experiment_run']);
  assert.equal(listed.result.tools[0].inputSchema.additionalProperties, false);
  const unconfigured = await client.rpc('tools/call', { name: 'experiment_run', arguments: { planPath: '/tmp/plan.json' } });
  assert.equal(JSON.parse(unconfigured.result.content[0].text).code, 'NOT_CONFIGURED');
  const extra = await client.rpc('tools/call', { name: 'experiment_run', arguments: { planPath: '/tmp/plan.json', command: '/bin/sh' } });
  assert.equal(JSON.parse(extra.result.content[0].text).code, 'INVALID_INPUT');
  const activation = await client.rpc('tools/call', { name: 'activate', arguments: { planPath: '/tmp/plan.json' } });
  assert.equal(JSON.parse(activation.result.content[0].text).code, 'UNKNOWN_TOOL');
  const relative = await client.rpc('tools/call', { name: 'experiment_run', arguments: { planPath: '../plan.json' } });
  assert.equal(JSON.parse(relative.result.content[0].text).code, 'INVALID_PATH');
  const invalidEnvelope = await client.rpc('tools/list', {}, { hiddenCommand: '/bin/sh' });
  assert.equal(invalidEnvelope.error.code, -32600);
});

test('real MCP engine lifecycle preserves fixture labels and repeated runs preserve prior artifacts', async t => {
  const temporary = await mkdtemp(join(tmpdir(), 'metaharness-wire-'));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const fixture = await createFixture({ directory: join(temporary, 'fixture'), binaryPath: join(repo, 'native/target/release/weight_evolution') });
  const planBefore = await readFile(fixture.planPath);
  const client = wire(fixture.configPath); t.after(() => client.close());
  const reply = await client.rpc('tools/call', { name: 'experiment_run', arguments: { planPath: fixture.planPath }, _meta: { progressToken: 'fixture' } });
  assert.equal(reply.error, undefined);
  assert.equal(reply.result.isError, false, JSON.stringify(reply));
  assert.deepEqual(JSON.parse(reply.result.content[0].text), reply.result.structuredContent);
  const report = reply.result.structuredContent;
  assert.equal(report.evidenceMode, 'fixture');
  assert.equal(report.selectedCandidateId, 'fixture-improvement');
  assert.equal(report.productionActivation, false);
  assert.equal(report.activationAuthorized, false);
  assert.equal(report.inferencePerformed, false);
  assert.equal(report.candidates.length, 2);
  assert(report.candidates.every(c => c.rollbackExact && c.rollbackVerified && c.rvfVerified));
  const reportBefore = await readFile(report.reportPath);
  assert.deepEqual(await readFile(fixture.planPath), planBefore);
  const replay = await client.rpc('tools/call', { name: 'experiment_run', arguments: { planPath: fixture.planPath } });
  assert.equal(replay.result.isError, false, JSON.stringify(replay));
  assert.notEqual(replay.result.structuredContent.runDirectory, report.runDirectory);
  assert.deepEqual(await readFile(report.reportPath), reportBefore, 'Repeated mutations must not overwrite prior artifacts');
});
