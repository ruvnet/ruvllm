#!/usr/bin/env node
// Fixed experiment engine; operator configuration never comes from a tool argument.
import { runExperiment, parseJson } from '../../../integrations/metaharness/src/engine.mjs';
import { tickLoop, readLoopStatus } from '../../../integrations/metaharness/src/gpu-loop.mjs';
import { loadHostAdapter } from '../../../integrations/metaharness/src/gpu-host.mjs';

const MAX_REQUEST = 65536;
const MAX_RESULT = 1048576;
const MAX_PENDING = 16;
export const TOOLS = [{
  name: 'experiment_run',
  description: 'Run a bounded local RuVLLM weight experiment from a frozen plan under an operator-pinned runtime. Writes candidates, rollback files and evidence to a private output root. Does not run model inference or activate a model. Synthetic fixtures remain explicitly labelled.',
  inputSchema: { type: 'object', properties: { planPath: { type: 'string', minLength: 1, maxLength: 4096, description: 'Absolute path to a plan inside operator-configured read roots.' } }, required: ['planPath'], additionalProperties: false },
  annotations: { title: 'Run bounded weight experiment', readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, {
  name: 'gpu_loop_status',
  description: 'Read the persisted operator-configured GPU loop status without contacting a provider or resolving secrets. Configuration cannot be supplied through tool arguments.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  annotations: { title: 'Read GPU loop status', readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
}, {
  name: 'gpu_loop_tick',
  description: 'Advance one durable GPU experiment loop tick through the operator-pinned host adapter. May submit, inspect or clean up GPU resources within the operator schedule and budget. The host resolves ruOS secret references; this tool never accepts or returns credentials. No model activation.',
  inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  annotations: { title: 'Advance scheduled GPU loop', readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
}];

function exact(value, required, optional = []) {
  return value && typeof value === 'object' && !Array.isArray(value) && required.every(k => Object.hasOwn(value, k)) && Object.keys(value).every(k => [...required, ...optional].includes(k));
}
function fault(code, message) { return { content: [{ type: 'text', text: JSON.stringify({ code, message }) }], isError: true }; }
function response(value) {
  const encoded = JSON.stringify(value);
  return Buffer.byteLength(encoded) > MAX_RESULT / 2
    ? fault('RESOURCE_LIMIT', 'Result exceeds wire bound; reconcile persisted state before retrying')
    : { content: [{ type: 'text', text: encoded }], structuredContent: value, isError: false };
}
export async function dispatch(message) {
  const hasId = message && Object.hasOwn(message, 'id');
  if (!exact(message, ['jsonrpc', 'method'], ['id', 'params']) || message.jsonrpc !== '2.0' || typeof message.method !== 'string' || (hasId && !(typeof message.id === 'string' || Number.isSafeInteger(message.id)))) {
    return { jsonrpc: '2.0', id: hasId && (typeof message.id === 'string' || Number.isSafeInteger(message.id)) ? message.id : null, error: { code: -32600, message: 'Invalid JSON-RPC request' } };
  }
  if (!hasId) return null;
  let result;
  if (message.method === 'initialize') result = { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'ruflo-metaharness', version: '0.2.0' } };
  else if (message.method === 'ping') result = {};
  else if (message.method === 'tools/list') result = { tools: TOOLS };
  else if (message.method === 'tools/call') {
    const params = message.params;
    if (!exact(params, ['name', 'arguments'], ['_meta'])) result = fault('INVALID_INPUT', 'Expected only name and arguments');
    else if (params.name === 'gpu_loop_status' || params.name === 'gpu_loop_tick') {
      if (!exact(params.arguments, [])) result = fault('INVALID_INPUT', 'GPU loop tools accept an empty arguments object only');
      else if (!process.env.RUVLLM_GPU_LOOP_CONFIG || (params.name === 'gpu_loop_tick' && (!process.env.RUVLLM_GPU_HOST_MODULE || !process.env.RUVLLM_GPU_HOST_SHA256))) result = fault('NOT_CONFIGURED', 'Set the operator GPU loop configuration and pinned host adapter bindings');
      else {
        try {
          const configPath = process.env.RUVLLM_GPU_LOOP_CONFIG;
          if (params.name === 'gpu_loop_status') result = response(await readLoopStatus({ configPath }));
          else {
            const adapter = await loadHostAdapter({ modulePath: process.env.RUVLLM_GPU_HOST_MODULE, sha256: process.env.RUVLLM_GPU_HOST_SHA256 });
            result = response(await tickLoop({ configPath, adapter }));
          }
        } catch { result = fault('GPU_LOOP_FAILED', 'GPU loop operation refused or failed; inspect operator configuration and reconcile persisted state'); }
      }
    }
    else if (params.name !== 'experiment_run') result = fault('UNKNOWN_TOOL', 'Unknown experiment tool');
    else if (!exact(params.arguments, ['planPath'])) result = fault('INVALID_INPUT', 'Expected only arguments.planPath');
    else if (typeof params.arguments.planPath !== 'string' || params.arguments.planPath.length > 4096 || !params.arguments.planPath.startsWith('/') || /[\x00-\x1f\x7f]/u.test(params.arguments.planPath)) result = fault('INVALID_PATH', 'A bounded absolute plan path is required');
    else if (!process.env.RUVLLM_EXPERIMENT_CONFIG) result = fault('NOT_CONFIGURED', 'Set RUVLLM_EXPERIMENT_CONFIG to an operator-owned runtime file');
    else {
      try {
        const report = await runExperiment({ configPath: process.env.RUVLLM_EXPERIMENT_CONFIG, planPath: params.arguments.planPath });
        result = response(report);
      } catch (error) { result = fault(typeof error.code === 'string' ? error.code : 'EXPERIMENT_FAILED', error.message || 'Experiment failed; reconcile output before retrying'); }
    }
  } else return { jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Method not found' } };
  return { jsonrpc: '2.0', id: message.id, result };
}

// A single serialized queue prevents concurrent mutations and bounds pending work.
let buffer = Buffer.alloc(0); let pending = 0; let queue = Promise.resolve();
function send(value) { if (value) process.stdout.write(`${JSON.stringify(value)}\n`); }
process.stdin.on('data', chunk => {
  buffer = Buffer.concat([buffer, chunk]);
  while (buffer.includes(10)) {
    const newline = buffer.indexOf(10); const line = buffer.subarray(0, newline); buffer = buffer.subarray(newline + 1);
    if (!line.length) continue;
    if (line.length > MAX_REQUEST || pending >= MAX_PENDING) { process.stderr.write('Experiment MCP request or queue bound exceeded\n'); process.exit(1); }
    pending++;
    queue = queue.then(async () => {
      let value;
      try { value = parseJson(line); }
      catch { send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } }); return; }
      send(await dispatch(value));
    }).catch(() => { send({ jsonrpc: '2.0', id: null, error: { code: -32603, message: 'Internal error; reconcile outputs before retrying' } }); }).finally(() => { pending--; });
  }
  if (buffer.length > MAX_REQUEST) { process.stderr.write('Experiment MCP request bound exceeded\n'); process.exit(1); }
});
process.stdin.on('end', () => { queue.finally(() => { if (buffer.length) process.exitCode = 1; }); });
