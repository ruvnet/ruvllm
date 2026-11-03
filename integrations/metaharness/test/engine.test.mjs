import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, chmod, symlink, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runExperiment, parseJson, fileDigest, sha256 } from '../src/engine.mjs';
import { createFixture } from '../src/fixture.mjs';

const binaryPath = process.env.RUFLO_NEURAL_TEST_BINARY ?? fileURLToPath(new URL('../../../native/target/release/weight_evolution', import.meta.url));
async function setup(t) {
  const temporary = await realpath(await mkdtemp(join(tmpdir(), 'ruvllm-engine-')));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const fixture = await createFixture({ directory: join(temporary, 'fixture'), binaryPath });
  return { ...fixture, temporary, plan: JSON.parse(await readFile(fixture.planPath)), config: JSON.parse(await readFile(fixture.configPath)) };
}
async function update(path, value) { await writeFile(path, JSON.stringify(value) + '\n'); }

test('actual native lifecycle selects evaluated fixture and preserves exact bytes', async t => {
  const fixture = await setup(t); const before = await fileDigest(fixture.plan.source);
  const report = await runExperiment(fixture);
  assert.equal(report.evidenceMode, 'fixture'); assert.equal(report.selectedCandidateId, 'fixture-improvement');
  assert.equal(report.productionActivation, false); assert.equal(report.activationAuthorized, false); assert.equal(report.inferencePerformed, false);
  assert.equal(report.sourceSha256, before); assert.equal(await fileDigest(fixture.plan.source), before);
  assert.equal(report.candidates.length, 2);
  for (const candidate of report.candidates) { assert.equal(candidate.rollbackExact, true); assert.equal(candidate.restoredSha256, before); assert.equal(candidate.rvfVerified, true); assert.match(candidate.rvfSha256, /^[a-f0-9]{64}$/); }
  assert.equal(report.candidates[0].status, 'qualified'); assert.equal(report.candidates[1].status, 'unqualified');
  assert.equal(report.developmentSelection.selectedId, 'fixture-improvement');
  assert.equal(report.developmentSelection.independentConfirmation, false);
  assert.equal(report.independentConfirmation, false);
  assert.deepEqual(report.candidates[1].gate.reasons, ['missing_evaluation']);
  assert.deepEqual(JSON.parse(await readFile(report.reportPath)), report);
  assert.equal(report.planSha256, sha256(await readFile(fixture.planPath)));
});

test('repeat runs use exclusive output directories', async t => {
  const fixture = await setup(t); const first = await runExperiment(fixture), second = await runExperiment(fixture);
  assert.notEqual(first.runDirectory, second.runDirectory); assert.equal(first.sourceSha256, second.sourceSha256);
});

test('policy substitution is rejected before execution', async t => {
  const fixture = await setup(t); await writeFile(fixture.plan.policy, '{}');
  await assert.rejects(runExperiment(fixture), /Policy is not approved/);
});

test('binary drift refuses all native execution', async t => {
  const fixture = await setup(t); fixture.config.binarySha256 = '0'.repeat(64); await update(fixture.configPath, fixture.config);
  await assert.rejects(runExperiment(fixture), /Binary digest mismatch/);
});

test('unknown plan fields and duplicate candidate IDs fail closed', async t => {
  const fixture = await setup(t); await update(fixture.planPath, { ...fixture.plan, activate: true });
  await assert.rejects(runExperiment(fixture), /Unknown object fields/);
  fixture.plan.candidates[1].id = fixture.plan.candidates[0].id; await update(fixture.planPath, fixture.plan);
  await assert.rejects(runExperiment(fixture), /duplicate candidate id/);
});

test('candidate limits and unsafe IDs are rejected', async t => {
  const fixture = await setup(t); fixture.config.maxCandidates = 1; await update(fixture.configPath, fixture.config);
  await assert.rejects(runExperiment(fixture), /Candidate limit/);
  fixture.config.maxCandidates = 16; await update(fixture.configPath, fixture.config);
  fixture.plan.candidates[0].id = '../escape'; await update(fixture.planPath, fixture.plan);
  await assert.rejects(runExperiment(fixture), /candidate id/);
});

test('outside roots and symlinked input files are rejected', async t => {
  const fixture = await setup(t); fixture.plan.candidates[0].patch = binaryPath; await update(fixture.planPath, fixture.plan);
  await assert.rejects(runExperiment(fixture), /outside configured/);
  const link = join(fixture.config.readRoots[0], 'linked.json'); await symlink(join(fixture.config.readRoots[0], 'patch.json'), link);
  fixture.plan.candidates[0].patch = link; await update(fixture.planPath, fixture.plan);
  await assert.rejects(runExperiment(fixture), /Symlinks/);
});

test('private config and output permissions are required', async t => {
  const fixture = await setup(t); await chmod(fixture.configPath, 0o644);
  await assert.rejects(runExperiment(fixture), /owner-only/);
  await chmod(fixture.configPath, 0o600); await chmod(fixture.config.writeRoot, 0o755);
  await assert.rejects(runExperiment(fixture), /Write root must be private/);
});

test('runtime binary and config cannot be mutable through outputs', async t => {
  const fixture = await setup(t); fixture.config.writeRoot = fixture.temporary; await chmod(fixture.temporary, 0o700); await update(fixture.configPath, fixture.config);
  await assert.rejects(runExperiment(fixture), /outside write root/);
});

test('protected edits reject and leave the source intact', async t => {
  const fixture = await setup(t); const policy = JSON.parse(await readFile(fixture.plan.policy));
  policy.protected = [{ tensor: 'layer.weight', start: 1, end: 2 }]; await update(fixture.plan.policy, policy);
  fixture.config.policySha256s = [sha256(await readFile(fixture.plan.policy))]; await update(fixture.configPath, fixture.config);
  const digest = await fileDigest(fixture.plan.source), report = await runExperiment(fixture);
  assert.equal(report.selectedCandidateId, null); assert.equal(await fileDigest(fixture.plan.source), digest);
  for (const candidate of report.candidates) { assert.equal(candidate.status, 'rejected'); assert.match(candidate.error, /protected scalar edit/); assert.equal(candidate.rollbackExact, false); }
});

test('forged evaluation bindings are rejected and exact rollback still runs', async t => {
  const fixture = await setup(t); const path = fixture.plan.candidates[0].evaluation, evaluation = JSON.parse(await readFile(path));
  evaluation.candidate.model_sha256 = '0'.repeat(64); await update(path, evaluation);
  const report = await runExperiment(fixture);
  assert.equal(report.selectedCandidateId, null); assert.equal(report.candidates[0].status, 'rejected'); assert.equal(report.candidates[0].rollbackExact, true);
});

test('regressed accuracy cannot qualify despite better completion', async t => {
  const fixture = await setup(t); const path = fixture.plan.candidates[0].evaluation, evaluation = JSON.parse(await readFile(path));
  evaluation.baseline_correct = 2; evaluation.candidate_correct = 1; await update(path, evaluation);
  const report = await runExperiment(fixture);
  assert.equal(report.selectedCandidateId, null); assert.equal(report.candidates[0].status, 'unqualified'); assert.equal(report.candidates[0].rollbackExact, true);
  assert.ok(report.candidates[0].gate.reasons.includes('primary_regressed'));
});

test('different evaluation suites cannot be compared across candidates', async t => {
  const fixture = await setup(t), original = fixture.plan.candidates[0].evaluation;
  const evaluation = JSON.parse(await readFile(original)); evaluation.suite_sha256 = '9'.repeat(64);
  const changed = join(fixture.config.readRoots[0], 'changed-suite.json'); await update(changed, evaluation);
  fixture.plan.candidates[1].evaluation = changed; await update(fixture.planPath, fixture.plan);
  const report = await runExperiment(fixture);
  assert.equal(report.candidates[1].status, 'rejected'); assert.match(report.candidates[1].error, /different baselines or suites/);
  assert.equal(report.candidates[1].rollbackExact, true); assert.equal(report.selectedCandidateId, 'fixture-improvement');
});

test('external evidence stays caller asserted and missing measurements never qualify', async t => {
  const fixture = await setup(t); fixture.plan.evidenceMode = 'external';
  fixture.plan.candidates = [fixture.plan.candidates[1]]; await update(fixture.planPath, fixture.plan);
  const report = await runExperiment(fixture);
  assert.equal(report.evidenceMode, 'external'); assert.equal(report.selectedCandidateId, null);
  assert.equal(report.inferencePerformed, false); assert.equal(report.independentConfirmation, false);
  assert.match(report.evaluationTrust, /caller-supplied/); assert.equal(report.candidates[0].status, 'unqualified');
});

test('oversized control files fail before candidate execution', async t => {
  const fixture = await setup(t); await writeFile(fixture.planPath, ' '.repeat(1024 * 1024 + 1));
  await assert.rejects(runExperiment(fixture), /Control file too large/);
});

test('untrusted JSON rejects duplicates, non-finite values and excessive nesting', () => {
  assert.throws(() => parseJson(Buffer.from('{"a":1,"a":2}')), /Duplicate/);
  assert.throws(() => parseJson(Buffer.from('{"a":1e400}')), /Non-finite/);
  assert.throws(() => parseJson(Buffer.from('['.repeat(100) + '0' + ']'.repeat(100))), /nesting/);
  assert.throws(() => parseJson(Buffer.from('true false')), /Trailing/);
  assert.equal(parseJson(Buffer.from('{"safe":[1,null,true,"ok"]}')).safe[3], 'ok');
});
