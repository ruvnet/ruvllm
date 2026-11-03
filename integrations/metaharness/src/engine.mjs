import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { constants } from 'node:fs';
import { open, lstat, realpath, mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname, join, relative, isAbsolute, parse } from 'node:path';
import { meetsPromotionRule, gateFingerprint } from '@metaharness/flywheel';
import { selectDevelopmentCandidate } from './adapters.mjs';

const MAX_JSON = 1024 * 1024;
const HASH = /^[a-f0-9]{64}$/;
const inside = (root, path) => path === root || (!relative(root, path).startsWith('..') && !isAbsolute(relative(root, path)));
const assert = (condition, message) => { if (!condition) throw new Error(message); };
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function keys(value, required, optional = []) {
  assert(value && typeof value === 'object' && !Array.isArray(value), 'Expected JSON object');
  assert(required.every(k => Object.hasOwn(value, k)), `Missing fields: ${required.join(', ')}`);
  assert(Object.keys(value).every(k => [...required, ...optional].includes(k)), 'Unknown object fields');
}

// Reject duplicate object keys and non-finite numbers before applying schemas.
export function parseJson(bytes) {
  const text = bytes.toString('utf8'); let index = 0;
  const ws = () => { while (/[ \t\r\n]/.test(text[index] ?? '') && index < text.length) index++; };
  function string() {
    const start = index++; let escaped = false;
    while (index < text.length) {
      const c = text[index++];
      if (c === '"' && !escaped) return JSON.parse(text.slice(start, index));
      if (c === '\\' && !escaped) escaped = true; else escaped = false;
    }
    throw new Error('Unterminated JSON string');
  }
  function value(depth = 0) {
    assert(depth < 64, 'JSON nesting limit'); ws();
    if (text[index] === '"') return string();
    if (text[index] === '{') {
      index++; ws(); const result = Object.create(null); const seen = new Set();
      if (text[index] === '}') { index++; return result; }
      while (true) {
        ws(); assert(text[index] === '"', 'Invalid JSON key'); const key = string();
        assert(!seen.has(key), 'Duplicate JSON key'); seen.add(key); ws();
        assert(text[index++] === ':', 'Invalid JSON separator'); result[key] = value(depth + 1); ws();
        const end = text[index++]; if (end === '}') return result; assert(end === ',', 'Invalid JSON object');
      }
    }
    if (text[index] === '[') {
      index++; ws(); const result = []; if (text[index] === ']') { index++; return result; }
      while (true) { result.push(value(depth + 1)); ws(); const end = text[index++]; if (end === ']') return result; assert(end === ',', 'Invalid JSON array'); }
    }
    const token = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(text.slice(index));
    assert(token, 'Invalid JSON value'); index += token[0].length; const result = JSON.parse(token[0]);
    assert(typeof result !== 'number' || Number.isFinite(result), 'Non-finite JSON number'); return result;
  }
  const result = value(); ws(); assert(index === text.length, 'Trailing JSON data'); return result;
}

export async function checkedPath(path, { directory = false, roots } = {}) {
  assert(typeof path === 'string' && isAbsolute(path) && !path.includes('\0'), 'Expected absolute path');
  assert(path === resolve(path), 'Path must be canonical without traversal');
  let cursor = parse(path).root;
  for (const part of path.slice(cursor.length).split('/').filter(Boolean)) {
    cursor = join(cursor, part); const stat = await lstat(cursor); assert(!stat.isSymbolicLink(), 'Symlinks are forbidden');
  }
  const canonical = await realpath(path); const stat = await lstat(canonical);
  assert(directory ? stat.isDirectory() : stat.isFile(), directory ? 'Expected directory' : 'Expected regular file');
  if (roots) assert(roots.some(root => inside(root, canonical)), 'Path outside configured read roots');
  return canonical;
}

async function boundedRead(path, limit = MAX_JSON) {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = await file.stat(); assert(stat.isFile() && stat.size <= limit, 'Control file too large or not regular');
    const bytes = Buffer.alloc(stat.size); const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
    assert(bytesRead === bytes.length, 'Control file changed during read'); return bytes;
  } finally { await file.close(); }
}

export async function fileDigest(path) {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    assert((await file.stat()).isFile(), 'Expected regular file');
    const hash = createHash('sha256'); const bytes = Buffer.alloc(1024 * 1024); let position = 0;
    while (true) { const { bytesRead } = await file.read(bytes, 0, bytes.length, position); if (!bytesRead) break; hash.update(bytes.subarray(0, bytesRead)); position += bytesRead; }
    return hash.digest('hex');
  } finally { await file.close(); }
}

async function identicalFiles(a, b) {
  const first = await open(a, constants.O_RDONLY | constants.O_NOFOLLOW), second = await open(b, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const size = (await first.stat()).size; if (size !== (await second.stat()).size) return false;
    const x = Buffer.alloc(Math.min(size, 1024 * 1024)), y = Buffer.alloc(x.length);
    for (let position = 0; position < size; position += x.length) {
      const length = Math.min(x.length, size - position);
      const [rx, ry] = await Promise.all([first.read(x, 0, length, position), second.read(y, 0, length, position)]);
      if (rx.bytesRead !== length || ry.bytesRead !== length || !x.subarray(0, length).equals(y.subarray(0, length))) return false;
    }
    return true;
  } finally { await first.close(); await second.close(); }
}

async function save(path, value) { await writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx', mode: 0o600 }); }

async function runtime(configPath) {
  configPath = await checkedPath(configPath); const bytes = await boundedRead(configPath); const config = parseJson(bytes);
  keys(config, ['schema', 'binaryPath', 'binarySha256', 'readRoots', 'writeRoot', 'policySha256s', 'maxCandidates', 'timeoutMs']);
  assert(config.schema === 'ruvllm-experiment-runtime/v1', 'Unsupported runtime schema');
  assert(HASH.test(config.binarySha256), 'Invalid binary digest');
  assert(Array.isArray(config.readRoots) && config.readRoots.length > 0 && config.readRoots.length <= 32, 'Invalid read roots');
  assert(Array.isArray(config.policySha256s) && config.policySha256s.length > 0 && config.policySha256s.length <= 128 && config.policySha256s.every(x => HASH.test(x)), 'Invalid approved policy digests');
  assert(Number.isInteger(config.maxCandidates) && config.maxCandidates >= 1 && config.maxCandidates <= 16, 'Candidate limit must be 1..16');
  assert(Number.isInteger(config.timeoutMs) && config.timeoutMs >= 100 && config.timeoutMs <= 300000, 'Invalid timeout');
  config.readRoots = await Promise.all(config.readRoots.map(path => checkedPath(path, { directory: true })));
  config.writeRoot = await checkedPath(config.writeRoot, { directory: true });
  config.binaryPath = await checkedPath(config.binaryPath);
  assert(!inside(config.writeRoot, configPath) && !inside(config.writeRoot, config.binaryPath), 'Runtime config and binary must be outside write root');
  const stat = await lstat(configPath); assert((stat.mode & 0o077) === 0, 'Runtime config must be owner-only');
  assert(((await lstat(config.writeRoot)).mode & 0o077) === 0, 'Write root must be private');
  assert(await fileDigest(config.binaryPath) === config.binarySha256, 'Binary digest mismatch');
  return { ...config, configSha256: sha256(bytes) };
}

async function native(config, request) {
  assert(await fileDigest(config.binaryPath) === config.binarySha256, 'Binary digest mismatch');
  const input = JSON.stringify(request); assert(Buffer.byteLength(input) <= 64 * 1024, 'Native request limit');
  return new Promise((accept, reject) => {
    const environment = Object.fromEntries(['PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL'].filter(k => process.env[k]).map(k => [k, process.env[k]]));
    const child = spawn(config.binaryPath, [], { stdio: ['pipe', 'pipe', 'pipe'], env: environment, shell: false });
    let stdout = [], stderr = [], outSize = 0, errSize = 0, failure;
    const fail = message => { failure ??= new Error(message); child.kill('SIGKILL'); };
    const timer = setTimeout(() => fail('Native operation timed out'), config.timeoutMs);
    child.stdout.on('data', bytes => { outSize += bytes.length; if (outSize > MAX_JSON) fail('Native stdout limit'); else stdout.push(bytes); });
    child.stderr.on('data', bytes => { errSize += bytes.length; if (errSize > 64 * 1024) fail('Native stderr limit'); else stderr.push(bytes); });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.stdin.on('error', () => {});
    child.on('close', code => {
      clearTimeout(timer); if (failure) return reject(failure);
      if (code !== 0) return reject(new Error(`Native ${request.op} rejected: ${Buffer.concat(stderr).toString('utf8').slice(0, 2048)}`));
      try { accept(parseJson(Buffer.concat(stdout))); } catch (error) { reject(error); }
    });
    child.stdin.end(input);
  });
}

export async function runExperiment({ configPath, planPath }) {
  const config = await runtime(configPath); planPath = await checkedPath(planPath, { roots: config.readRoots });
  const planBytes = await boundedRead(planPath), plan = parseJson(planBytes);
  keys(plan, ['schema', 'evidenceMode', 'source', 'policy', 'candidates']);
  assert(plan.schema === 'ruvllm-experiment-plan/v1', 'Unsupported plan schema');
  assert(['fixture', 'external'].includes(plan.evidenceMode), 'Explicit fixture or external evidence mode required');
  assert(Array.isArray(plan.candidates) && plan.candidates.length >= 1 && plan.candidates.length <= config.maxCandidates, 'Candidate limit exceeded');
  const source = await checkedPath(plan.source, { roots: config.readRoots });
  const policyPath = await checkedPath(plan.policy, { roots: config.readRoots });
  const policyBytes = await boundedRead(policyPath), policy = parseJson(policyBytes), policySha256 = sha256(policyBytes);
  assert(config.policySha256s.includes(policySha256), 'Policy is not approved');
  assert(Array.isArray(policy.allowed_tensors) && policy.allowed_tensors.length > 0 && policy.allowed_tensors.length <= 256, 'Invalid tensor selection');
  const ids = new Set(), candidates = [];
  for (const candidate of plan.candidates) {
    keys(candidate, ['id', 'patch'], ['evaluation']);
    assert(typeof candidate.id === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(candidate.id) && !ids.has(candidate.id), 'Invalid or duplicate candidate id'); ids.add(candidate.id);
    const patchPath = await checkedPath(candidate.patch, { roots: config.readRoots }), patchBytes = await boundedRead(patchPath);
    const entry = { ...candidate, patchBytes, patch: parseJson(patchBytes) };
    if (candidate.evaluation) { const path = await checkedPath(candidate.evaluation, { roots: config.readRoots }); entry.evaluationBytes = await boundedRead(path); entry.evaluationValue = parseJson(entry.evaluationBytes); }
    candidates.push(entry);
  }
  const runDirectory = join(config.writeRoot, `run-${randomUUID()}`); await mkdir(runDirectory, { mode: 0o700 });
  await writeFile(join(runDirectory, 'plan.json'), planBytes, { flag: 'wx', mode: 0o600 });
  const copiedPolicy = join(runDirectory, 'policy.json'); await writeFile(copiedPolicy, policyBytes, { flag: 'wx', mode: 0o600 });
  const inspection = await native(config, { op: 'inspect', source, tensors: policy.allowed_tensors });
  const reportPath = join(runDirectory, 'report.json');
  const report = { schema: 'ruvllm-experiment-report/v1', evidenceMode: plan.evidenceMode, sourceSha256: inspection.source_sha256,
    policySha256, planSha256: sha256(planBytes), binarySha256: config.binarySha256, configSha256: config.configSha256,
    gateSha256: gateFingerprint(meetsPromotionRule), runDirectory, reportPath, productionActivation: false, activationAuthorized: false,
    inferencePerformed: false, independentConfirmation: false, evaluationTrust: 'caller-supplied; not independent attestation', costMetered: false, candidates: [], selectedCandidateId: null };
  let sharedEvaluation, baselineScore;
  for (const candidate of candidates) {
    const output = join(runDirectory, candidate.id); await mkdir(output, { mode: 0o700 });
    const patch = join(output, 'patch.json'), checkpoint = join(output, 'candidate.safetensors'), receiptPath = join(output, 'receipt.json');
    await writeFile(patch, candidate.patchBytes, { flag: 'wx', mode: 0o600 });
    const result = { id: candidate.id, status: 'rejected', patchSha256: sha256(candidate.patchBytes), rollbackExact: false, rollbackVerified: false, rvfVerified: false, productionActivation: false };
    let receipt;
    try {
      assert(candidate.patch.source_sha256 === report.sourceSha256, 'Patch source mismatch');
      receipt = await native(config, { op: 'patch', source, output: checkpoint, patch, policy: copiedPolicy });
      assert(receipt.source_sha256 === report.sourceSha256 && receipt.policy_sha256 === policySha256 && receipt.patch_sha256 === result.patchSha256 && receipt.activation_authorized === false, 'Native receipt binding mismatch');
      await save(receiptPath, receipt); result.candidateSha256 = receipt.candidate_sha256; result.status = 'unqualified';
      let evaluationPath;
      if (candidate.evaluationBytes) {
        evaluationPath = join(output, 'evaluation.json'); await writeFile(evaluationPath, candidate.evaluationBytes, { flag: 'wx', mode: 0o600 });
        const value = candidate.evaluationValue;
        result.evaluationSha256 = sha256(candidate.evaluationBytes);
        // All candidates must use the same suite, baseline, runtime and protocol.
        const signature = JSON.stringify([value.suite_sha256, value.protocol_sha256, value.baseline, value.total, value.baseline_correct, value.baseline_truncated, value.baseline_output_sha256]);
        const validation = await native(config, { op: 'evaluate', evaluation: evaluationPath, receipt: receiptPath });
        assert(validation.valid === true && validation.activation_authorized === false, 'Evaluation not valid');
        if (sharedEvaluation) assert(sharedEvaluation === signature, 'Candidates evaluated against different baselines or suites'); else sharedEvaluation = signature;
        const total = value.total;
        assert(Number.isInteger(total) && total > 0, 'Evaluation total must be positive');
        const score = (correct, truncated) => ({ primary: correct / total, noopRate: truncated / total, costPerWin: 0, regressed: false });
        baselineScore = value.baseline_correct / total;
        const gate = meetsPromotionRule({ baseline: score(value.baseline_correct, value.baseline_truncated), candidate: score(value.candidate_correct, value.candidate_truncated) });
        if (value.candidate_correct <= value.baseline_correct) { gate.promote = false; gate.reasons.push('no_strict_accuracy_improvement'); }
        result.gate = { ...gate, meaning: 'research qualification only; no activation authority' };
        result.score = score(value.candidate_correct, value.candidate_truncated);
        result.status = gate.promote ? 'qualified' : 'unqualified';
      } else result.gate = { promote: false, reasons: ['missing_evaluation'] };
      const artifact = join(output, 'evidence.rvf');
      const exported = await native(config, { op: 'export', patch, policy: copiedPolicy, receipt: receiptPath, output: artifact, ...(evaluationPath ? { evaluation: evaluationPath } : {}) });
      const verified = await native(config, { op: 'verify', artifact, expected_sha256: exported.sha256 });
      assert(verified.integrity_verified === true && verified.activation_authorized === false, 'RVF verification failed'); result.rvfSha256 = exported.sha256; result.rvfVerified = true;
    } catch (error) { result.status = 'rejected'; result.error = error.message; }
    finally {
      if (receipt) {
        const restored = join(output, 'restored.safetensors');
        const rollback = await native(config, { op: 'rollback', source: checkpoint, output: restored, patch, policy: copiedPolicy, receipt: receiptPath });
        result.restoredSha256 = rollback.candidate_sha256;
        result.rollbackExact = result.rollbackVerified = rollback.candidate_sha256 === report.sourceSha256 && await identicalFiles(source, restored);
        assert(result.rollbackExact, 'Rollback did not restore original bytes');
      }
    }
    report.candidates.push(result);
  }
  assert(await fileDigest(source) === report.sourceSha256, 'Source checkpoint changed during experiment');
  const qualified = report.candidates.filter(candidate => candidate.status === 'qualified');
  qualified.sort((a, b) => b.score.primary - a.score.primary || a.score.noopRate - b.score.noopRate || a.id.localeCompare(b.id));
  if (qualified.length) {
    report.developmentSelection = await selectDevelopmentCandidate({
      baseline: { id: 'source.baseline', score: baselineScore, evidenceSha256: report.sourceSha256 },
      candidates: qualified.map(candidate => ({ id: candidate.id, score: candidate.score.primary, evidenceSha256: candidate.evaluationSha256 })),
      envelope: Object.fromEntries(['sourceSha256', 'policySha256', 'binarySha256', 'gateSha256', 'planSha256'].map(key => [key, report[key]])),
    });
    assert(qualified.some(candidate => candidate.id === report.developmentSelection.selectedId), 'Darwin selected an unqualified candidate');
    report.selectedCandidateId = report.developmentSelection.selectedId;
  }
  await save(reportPath, report); return report;
}
