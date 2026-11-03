import { test } from 'node:test';
import assert from 'node:assert/strict';
import { exportArchive, trainingPlan, selectDevelopmentCandidate } from '../src/adapters.mjs';

const trajectory = (id, resolved, tier = 'cheap', model = 'qwen') => ({
  instance_id: id, resolved, tier, model, model_patch: resolved ? 'diff --git a/x b/x' : '',
  messages: [{ role: 'system', content: 'Repair the issue' }, { role: 'user', content: id },
    { role: 'assistant', content: resolved ? 'Fixed issue' : 'Unable to fix' }],
});
test('Weight-EFT excludes holdouts and keeps same-model preference pairs', () => {
  const out = exportArchive({ archive: [trajectory('train-1', true), trajectory('train-1', false),
    trajectory('eval-1', true), trajectory('train-2', true, 'frontier', 'teacher')], evalHoldout: ['eval-1'] });
  assert.equal(out.sft.length, 2);
  assert.equal(out.dpo.length, 1);
  assert.equal(out.report.excludedByHoldout, 1);
  assert.equal(out.trainingExecuted, false);
  assert.equal(out.productionActivation, false);
});
test('Weight-EFT filters reward-hacked success', () => {
  const row = trajectory('hack-1', true);
  row.messages.push({ role: 'assistant', content: null, tool_calls: [{ id: 'call-1', type: 'function',
    function: { name: 'bash', arguments: JSON.stringify({ command: 'cat /tmp/gold.patch' }) } }] });
  const out = exportArchive({ archive: [row], evalHoldout: [] });
  assert.equal(out.report.droppedRewardHacked, 1);
  assert.equal(out.sft.length, 0);
});
test('archive budgets and boolean outcomes fail closed', () => {
  assert.throws(() => exportArchive({ archive: [{ ...trajectory('train-1', true), resolved: 'true' }], evalHoldout: [] }));
  assert.throws(() => exportArchive({ archive: [], maxTokens: Infinity }));
});
test('training plan never claims upstream dry run executed training', () => {
  const out = trainingPlan({ base: { id: 'Qwen/Qwen3.5-4B', paramsB: 4 },
    sftPath: '/tmp/sft.jsonl', dpoPath: '/tmp/dpo.jsonl', adapterPrefix: '/tmp/adapter' });
  assert.equal(out.executionStatus, 'not_executed');
  assert.equal(out.runnerCompatibility, 'adapter-required');
});
test('Darwin uses actual contrastive selection with immutable envelope and measured evidence', async () => {
  const envelope = { sourceSha256: 'a'.repeat(64), policySha256: 'b'.repeat(64) };
  const row = (id, score) => ({ id, score, evidenceSha256: 'c'.repeat(64) });
  const out = await selectDevelopmentCandidate({ baseline: row('base', 0.4),
    candidates: [row('lower', 0.2), row('better', 0.8)], envelope });
  assert.equal(out.selectedId, 'better');
  assert.equal(out.result.outcome, 'candidate');
  assert.equal(out.productionActivation, false);
  assert.deepEqual(envelope, { sourceSha256: 'a'.repeat(64), policySha256: 'b'.repeat(64) });
});
test('Darwin retains baseline on regressions and rejects nonfinite evidence', async () => {
  const row = (id, score) => ({ id, score, evidenceSha256: 'c'.repeat(64) });
  const out = await selectDevelopmentCandidate({ baseline: row('base', 0.9), candidates: [row('bad', 0.5)], envelope: {} });
  assert.equal(out.selectedId, 'base');
  await assert.rejects(selectDevelopmentCandidate({ baseline: row('base', 0.9), candidates: [row('bad', NaN)], envelope: {} }));
});
