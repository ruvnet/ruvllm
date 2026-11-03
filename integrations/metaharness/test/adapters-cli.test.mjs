import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const cli = fileURLToPath(new URL('../src/adapters-cli.mjs', import.meta.url));
const run = (operation, input) => spawnSync(process.execPath, [cli, operation], {
  input: typeof input === 'string' ? input : JSON.stringify(input), encoding: 'utf8', timeout: 10000,
});
test('JSON CLI exports standard datasets and reports explicit holdout provenance', () => {
  const result = run('export', { archive: [], evalHoldout: ['reserved-evaluation'] });
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.deepEqual(output.sft, []);
  assert.deepEqual(output.dpo, []);
  assert.equal(output.trainingExecuted, false);
  assert.match(output.holdoutSha256, /^[a-f0-9]{64}$/);
});
test('JSON CLI refuses malformed input and unknown commands with nonzero status', () => {
  assert.equal(run('export', '{').status, 1);
  assert.equal(run('activate', '{}').status, 2);
});
test('JSON CLI never turns a training plan into a training completion', () => {
  const result = run('training-plan', { base: { id: 'Qwen/Qwen3.5-4B', paramsB: 4 },
    sftPath: '/tmp/sft.jsonl', dpoPath: '/tmp/dpo.jsonl', adapterPrefix: '/tmp/candidate' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).executionStatus, 'not_executed');
});
