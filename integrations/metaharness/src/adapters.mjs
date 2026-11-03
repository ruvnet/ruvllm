// MetaHarness adapters are data-only: no provider requests, trainer launch or activation.
import { createHash } from 'node:crypto';
import { exportTrainingData, twoStagePlan } from '@metaharness/weight-eft';
import { runContrastive, digestDocument } from '@metaharness/darwin/contrastive';

const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const require = (condition, message) => { if (!condition) throw new Error(message); };
const identifier = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value);

/** Export reviewed caller-asserted trajectories; held-out instance IDs are always excluded. */
export function exportArchive({ archive, evalHoldout, maxTokens = 12000 }) {
  require(Array.isArray(archive) && archive.length <= 10000, 'Archive limit is 10000 trajectories');
  require(Array.isArray(evalHoldout) && evalHoldout.every(identifier), 'Explicit eval holdout IDs required');
  require(Number.isInteger(maxTokens) && maxTokens >= 1 && maxTokens <= 32768, 'Invalid token budget');
  require(Buffer.byteLength(JSON.stringify(archive)) <= 16 * 1024 * 1024, 'Archive exceeds 16 MiB');
  for (const row of archive) {
    require(row && identifier(row.instance_id) && typeof row.model === 'string' && row.model.length > 0,
      'Invalid trajectory identity');
    require(['cheap', 'frontier'].includes(row.tier) && typeof row.resolved === 'boolean', 'Invalid trajectory outcome');
    require(typeof row.model_patch === 'string' && Array.isArray(row.messages) && row.messages.length <= 4096,
      'Invalid trajectory content');
    for (const message of row.messages) {
      require(message && ['system', 'user', 'assistant', 'tool'].includes(message.role)
        && (typeof message.content === 'string' || message.content === null), 'Invalid message');
    }
  }
  const result = exportTrainingData(archive, {
    evalHoldout, maxTokens, truncateOverLength: false, dropRewardHacked: true,
  });
  return {
    schema: 'ruvllm-training-export/v1', ...result,
    archiveSha256: hash(archive), holdoutSha256: hash(evalHoldout),
    evidenceOrigin: 'caller-asserted-trajectories', trainingExecuted: false,
    productionActivation: false,
  };
}

/** Upstream command is informational; this checkout has no `ruvllm microlora` CLI. */
export function trainingPlan({ base, sftPath, dpoPath, adapterPrefix }) {
  require(base && typeof base.id === 'string' && Number.isFinite(base.paramsB), 'Invalid model');
  for (const path of [sftPath, dpoPath, adapterPrefix]) require(typeof path === 'string' && path.length > 0, 'Path required');
  return {
    schema: 'ruvllm-training-plan/v1',
    plans: twoStagePlan(base, sftPath, dpoPath, adapterPrefix),
    executionStatus: 'not_executed',
    runnerCompatibility: 'adapter-required',
    reason: 'The upstream ruvllm microlora command is not implemented by this standalone checkout. Use a separately validated trainer; DPO requires an additional trainer implementation.',
    productionActivation: false,
  };
}

/** Replay bounded measured scores through Darwin. This ranks development evidence only. */
export async function selectDevelopmentCandidate({ baseline, candidates, envelope }) {
  require(envelope && typeof envelope === 'object' && !Array.isArray(envelope), 'Immutable experiment envelope required');
  require(Array.isArray(candidates) && candidates.length > 0 && candidates.length <= 32, 'Expected 1..32 candidates');
  const all = [baseline, ...candidates];
  for (const candidate of all) {
    require(candidate && identifier(candidate.id) && Number.isFinite(candidate.score)
      && candidate.score >= 0 && candidate.score <= 1
      && /^[a-f0-9]{64}$/.test(candidate.evidenceSha256), 'Invalid measured candidate');
  }
  require(new Set(all.map(row => row.id)).size === all.length, 'Duplicate candidate ID');
  const byId = new Map(all.map(row => [row.id, { ...row }]));
  const best = candidates.reduce((a, b) => b.score > a.score ? b : a);
  const document = { components: [{ id: 'weight-candidate', scope: 'domain', content: { id: baseline.id } }] };
  const result = await runContrastive({
    tenantId: 'local', projectId: 'ruvllm', task: { purpose: 'development-candidate-selection' },
    environment: { mode: 'evidence-replay' },
    library: { tenantId: 'local', projectId: 'ruvllm', entries: [] },
    envelope, baseline: document,
    budget: { maxUsd: 0, maxCalls: 5, timeoutMs: 5000, probeUsd: 0, proposeUsd: 0, composeUsd: 0 },
    evaluator: {
      surfaceAware: true,
      async probe({ candidate }) {
        const row = byId.get(candidate.components[0].content.id);
        require(row, 'Unmeasured candidate');
        return { appliedDigest: digestDocument(candidate), score: row.score, costUsd: 0,
          observations: [{ id: 'measured-failure', componentId: 'weight-candidate', message: `Bound evidence ${row.evidenceSha256}` }] };
      },
    },
    proposer: {
      async propose({ scope }) {
        return { edits: scope === 'domain' ? [{ componentId: 'weight-candidate', content: { id: best.id },
          citations: ['measured-failure'] }] : [], costUsd: 0 };
      },
    },
  });
  return { schema: 'ruvllm-development-selection/v1', selectedId: result.candidate.components[0].content.id,
    envelopeSha256: hash(envelope), evidenceSha256: hash(all), evidenceMode: 'score-replay',
    result, independentConfirmation: false, productionActivation: false };
}
