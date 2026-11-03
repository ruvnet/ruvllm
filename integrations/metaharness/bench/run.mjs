#!/usr/bin/env node
// CPU orchestration benchmark. Synthetic outcomes are contract fixtures, never inference.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { availableParallelism, cpus, platform, arch, release, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { runExperiment } from '../src/engine.mjs';
import { createFixture } from '../src/fixture.mjs';

const execute = promisify(execFile);
const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const percentile = (values, fraction) => {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
};

function parseArguments(argv) {
  const opts = { iterations: 5, warmup: 1, candidates: 2,
    binary: resolve(packageRoot, '../../native/target/release/weight_evolution') };
  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index];
    if (option === '--help') return { help: true };
    if (!['--iterations', '--warmup', '--candidates', '--binary', '--output'].includes(option)) {
      throw new Error(`Unknown option: ${option}`);
    }
    const value = argv[++index];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${option}`);
    opts[option.slice(2)] = ['--binary', '--output'].includes(option) ? resolve(value) : Number(value);
  }
  for (const [key, minimum, maximum] of [['iterations', 1, 100], ['warmup', 0, 10], ['candidates', 2, 16]]) {
    if (!Number.isInteger(opts[key]) || opts[key] < minimum || opts[key] > maximum) {
      throw new Error(`${key} must be an integer in [${minimum}, ${maximum}]`);
    }
  }
  return opts;
}

async function validateReport(report, plan, originalBytes, binarySha256) {
  assert.equal(report.schema, 'ruvllm-experiment-report/v1');
  assert.equal(report.evidenceMode, 'fixture');
  assert.equal(report.productionActivation, false);
  assert.equal(report.inferencePerformed, false);
  assert.equal(report.binarySha256, binarySha256);
  assert.equal(report.sourceSha256, sha256(originalBytes));
  assert.equal(report.candidates.length, plan.candidates.length);
  assert.equal(typeof report.selectedCandidateId, 'string', 'Fixture must qualify at least one candidate');
  assert(plan.candidates.some(({ id }) => id === report.selectedCandidateId));
  assert.equal(report.developmentSelection.schema, 'ruvllm-development-selection/v1');
  assert.equal(report.developmentSelection.selectedId, report.selectedCandidateId);
  assert.equal(report.developmentSelection.independentConfirmation, false);
  for (const candidate of report.candidates) {
    assert.equal(candidate.status, 'qualified', `Unexpected fixture outcome for ${candidate.id}`);
    assert.equal(candidate.rollbackExact, true, 'Rollback must be byte exact');
    assert.equal(candidate.restoredSha256, report.sourceSha256);
    assert.match(candidate.rvfSha256, /^[0-9a-f]{64}$/);
    assert.equal(candidate.gate.promote, true);
  }
  assert.deepEqual(await readFile(plan.source), originalBytes, 'Source checkpoint changed');
  const persisted = JSON.parse(await readFile(report.reportPath, 'utf8'));
  assert.equal(persisted.planSha256, report.planSha256, 'Persisted report does not match this run');
  assert.equal(persisted.runDirectory, report.runDirectory);
  return {
    reportSha256: sha256(await readFile(report.reportPath)),
    sourceSha256: report.sourceSha256,
    planSha256: report.planSha256,
    candidates: report.candidates.map(({ id, candidateSha256, restoredSha256, rollbackExact, rvfSha256 }) =>
      ({ id, candidateSha256, restoredSha256, rollbackExact, rvfSha256 })),
  };
}

export async function benchmark(options) {
  const root = await mkdtemp(join(tmpdir(), 'ruvllm-metaharness-bench-'));
  try {
    const fixture = await createFixture({ directory: join(root, 'fixture'), binaryPath: options.binary });
    const plan = JSON.parse(await readFile(fixture.planPath, 'utf8'));
    assert(plan.candidates.length > 0, 'Empty fixture plan');
    const seed = plan.candidates[0];
    // Same precomputed synthetic edit/evaluation repeated under distinct IDs. This measures
    // orchestration throughput, not proposal diversity or discovery of a better intervention.
    plan.candidates = Array.from({ length: options.candidates }, (_, index) =>
      ({ ...seed, id: `candidate-${index + 1}` }));
    await writeFile(fixture.planPath, `${JSON.stringify(plan, null, 2)}\n`);
    const originalBytes = await readFile(plan.source);
    const binarySha256 = sha256(await readFile(options.binary));
    const samples = { freshProcess: [], reusedProcess: [] };
    const evidence = { freshProcess: [], reusedProcess: [] };
    const invoke = async (mode) => {
      const started = performance.now();
      const report = mode === 'reusedProcess'
        ? await runExperiment(fixture)
        : JSON.parse((await execute(process.execPath,
          [join(packageRoot, 'src/cli.mjs'), 'run', '--config', fixture.configPath, '--plan', fixture.planPath],
          { maxBuffer: 8 * 1024 * 1024, timeout: 120_000 })).stdout);
      // Common benchmark-side assertions are outside the timer; all runner/native checks are inside.
      const elapsedMs = performance.now() - started;
      const verified = await validateReport(report, plan, originalBytes, binarySha256);
      return { elapsedMs, verified };
    };
    for (let i = 0; i < options.warmup; i += 1) {
      await invoke('freshProcess');
      await invoke('reusedProcess');
    }
    // Alternate the order to reduce systematic ordering effects from filesystem caches.
    for (let index = 0; index < options.iterations; index += 1) {
      for (const mode of index % 2 ? ['reusedProcess', 'freshProcess'] : ['freshProcess', 'reusedProcess']) {
        const sample = await invoke(mode);
        samples[mode].push(sample.elapsedMs);
        evidence[mode].push(sample.verified);
      }
    }
    const summarize = (values) => {
      const totalMs = values.reduce((a, b) => a + b, 0);
      return { samplesMs: values, p50Ms: percentile(values, 0.5), p95Ms: percentile(values, 0.95),
        meanMs: totalMs / values.length, experimentsPerSecond: values.length * 1000 / totalMs,
        candidatesPerSecond: values.length * options.candidates * 1000 / totalMs };
    };
    const freshProcess = summarize(samples.freshProcess);
    const reusedProcess = summarize(samples.reusedProcess);
    return {
      schema: 'ruvllm-metaharness-cpu-benchmark/v1',
      createdAt: new Date().toISOString(), evidenceMode: 'fixture', inferencePerformed: false,
      productionActivation: false,
      workload: { iterationsPerMode: options.iterations, warmupPerMode: options.warmup,
        candidatesPerExperiment: options.candidates, checkpointBytes: originalBytes.length,
        identicalCandidateEdits: true,
        description: 'Native inspect, bounded patch, supplied synthetic evaluation, RVF export/verification and exact rollback for each candidate, followed by Darwin development selection' },
      environment: { node: process.version, platform: platform(), architecture: arch(),
        kernel: release(), cpu: cpus()[0]?.model ?? 'unknown', parallelism: availableParallelism() },
      provenance: { binarySha256, benchmarkSourceSha256: sha256(await readFile(fileURLToPath(import.meta.url))),
        engineSourceSha256: sha256(await readFile(join(packageRoot, 'src/engine.mjs'))),
        fixtureSourceSha256: sha256(await readFile(join(packageRoot, 'src/fixture.mjs'))),
        cliSourceSha256: sha256(await readFile(join(packageRoot, 'src/cli.mjs'))),
        adaptersSourceSha256: sha256(await readFile(join(packageRoot, 'src/adapters.mjs'))),
        lockfileSha256: sha256(await readFile(join(packageRoot, 'package-lock.json'))) },
      freshProcess, reusedProcess,
      comparison: { p50ReductionPercent: 100 * (freshProcess.p50Ms - reusedProcess.p50Ms) / freshProcess.p50Ms,
        interpretation: 'Reusing Node avoids process startup/module loading; every experiment still executes the same native checks. Warm filesystem caches, tiny fixtures and small sample counts limit generalization.' },
      assertions: { sourceUnchanged: true, allRollbacksExact: true, allFixtureCandidatesQualified: true,
        reportBindingsVerified: true, runnerChecksIdentical: true }, evidence,
    };
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2));
    if (options.help) {
      console.log('Usage: node bench/run.mjs [--iterations 5] [--warmup 1] [--candidates 2] [--binary PATH] [--output PATH]');
    } else {
      const result = await benchmark(options);
      const serialized = `${JSON.stringify(result, null, 2)}\n`;
      if (options.output) await writeFile(options.output, serialized, { flag: 'wx', mode: 0o600 });
      process.stdout.write(serialized);
    }
  } catch (error) {
    console.error(error.stack ?? error.message);
    process.exitCode = 1;
  }
}
