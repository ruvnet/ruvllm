#!/usr/bin/env node
// Fixed native executable, path-only tool arguments, no provider or activation API.
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { constants } from 'node:fs';
import { open, lstat, realpath, writeFile } from 'node:fs/promises';
import { dirname, basename, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const MAX_REQUEST = 65536;
const MAX_RESULT = 1048576;
const HASH = /^[0-9a-f]{64}$/;
const NAME = 'ruflo-neural-evolution';
const VERSION = '0.1.0';
const pathSchema = { type: 'string', minLength: 1, maxLength: 4096, description: 'Absolute local path within operator configured roots.' };
const properties = (fields) => Object.fromEntries(fields.map((field) => [field, pathSchema]));
const definitions = [
  ['weight_inspect', 'Inspect selected native checkpoint tensors. tensorList is a JSON file containing an array of tensor names. Activation statistics are not available.', ['source', 'tensorList'], [], true],
  ['weight_patch', 'Apply an approved bounded reversible patch to a new output checkpoint and save its receipt. This never activates the model.', ['source', 'output', 'patch', 'policy', 'receiptOutput'], [], false],
  ['weight_evaluate', 'Validate an existing evaluation receipt against a native patch receipt. This does not run inference, authenticate evaluator claims, qualify or activate a model.', ['evaluation', 'receipt'], [], true],
  ['weight_export', 'Package a patch, approved policy and receipts in a complete RVF container. Integrity never grants activation authority.', ['patch', 'policy', 'receipt', 'output'], ['evaluation', 'governanceEvidence'], false],
  ['weight_rollback', 'Restore the exact original checkpoint bytes into a new output file using the native patch receipt. No serving pointer is changed.', ['source', 'output', 'patch', 'policy', 'receipt', 'receiptOutput'], [], false],
  ['weight_verify', 'Verify complete RVF integrity against expectedDigestFile containing exactly {"sha256":"..."}. This does not authenticate governance evidence or authorize activation.', ['artifact', 'expectedDigestFile'], [], true],
];
export const TOOLS = definitions.map(([name, description, required, optional, readOnly]) => ({
  name, description, inputSchema: { type: 'object', properties: properties([...required, ...optional]), required, additionalProperties: false },
  annotations: { title: name, readOnlyHint: readOnly, destructiveHint: false, idempotentHint: readOnly, openWorldHint: false },
}));

class BoundaryError extends Error { constructor(code, message) { super(message); this.code = code; } }
const fail = (code, message) => { throw new BoundaryError(code, message); };
const object = (value) => { if (!value || typeof value !== 'object' || Array.isArray(value)) fail('INVALID_INPUT', 'Expected an object'); return value; };
const exact = (value, required, optional = []) => { object(value); if (required.some((k) => !Object.hasOwn(value, k)) || Object.keys(value).some((k) => ![...required, ...optional].includes(k))) fail('INVALID_INPUT', 'Missing or unknown fields'); };
const within = (root, path) => { const r = relative(root, path); return r === '' || (!r.startsWith(`..${sep}`) && r !== '..' && !isAbsolute(r)); };
const absolute = (value) => { if (typeof value !== 'string' || !isAbsolute(value) || value.length > 4096 || /[\x00-\x1f\x7f]/u.test(value)) fail('INVALID_PATH', 'An absolute local path is required'); return resolve(value); };

async function fileHandle(path, limit) {
  const flags = constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK;
  const handle = await open(path, flags);
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > limit) fail('INVALID_FILE', 'A bounded regular file is required');
    return { handle, stat };
  } catch (error) { await handle.close(); throw error; }
}
async function bytes(path, limit) {
  const { handle } = await fileHandle(path, limit);
  try {
    const result = Buffer.alloc(limit + 1);
    let length = 0;
    while (length <= limit) {
      const { bytesRead } = await handle.read(result, length, result.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    if (length > limit) fail('RESOURCE_LIMIT', 'File grew beyond its bound');
    return result.subarray(0, length);
  } finally { await handle.close(); }
}
async function fileDigest(path, limit) {
  const { handle } = await fileHandle(path, limit);
  try {
    const hash = createHash('sha256'); const buffer = Buffer.alloc(1024 * 1024); let total = 0;
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (!bytesRead) break;
      total += bytesRead;
      if (total > limit) fail('RESOURCE_LIMIT', 'File grew beyond its bound');
      hash.update(buffer.subarray(0, bytesRead));
    }
    return hash.digest('hex');
  } finally { await handle.close(); }
}
async function privateFile(path, limit) {
  const metadata = await lstat(path);
  if (metadata.isSymbolicLink() || (metadata.mode & 0o077) !== 0 || (process.getuid && metadata.uid !== process.getuid())) fail('CONFIG_PERMISSIONS', 'Configuration must be an owner only regular file');
  return bytes(path, limit);
}
export async function loadConfig(path = process.env.RUFLO_NEURAL_CONFIG) {
  if (!path) fail('NOT_CONFIGURED', 'Set RUFLO_NEURAL_CONFIG to an operator owned runtime configuration');
  if (constants.O_NOFOLLOW === undefined || constants.O_NONBLOCK === undefined) fail('UNSUPPORTED_PLATFORM', 'Opened handle validation requires a supported Unix runtime');
  const configPath = absolute(path);
  const config = JSON.parse((await privateFile(configPath, 65536)).toString('utf8'));
  exact(config, ['schema', 'binaryPath', 'binarySha256', 'readRoots', 'writeRoot', 'policySha256s'], ['timeoutMs']);
  if (config.schema !== 'ruflo-neural-runtime/v1' || typeof config.binarySha256 !== 'string' || !HASH.test(config.binarySha256)) fail('INVALID_CONFIG', 'Invalid schema or executable digest');
  if (!Array.isArray(config.readRoots) || config.readRoots.length < 1 || config.readRoots.length > 16 || !Array.isArray(config.policySha256s) || config.policySha256s.length > 32 || config.policySha256s.some((v) => typeof v !== 'string' || !HASH.test(v))) fail('INVALID_CONFIG', 'Bounded read roots and policy digests required');
  const binaryPath = await realpath(absolute(config.binaryPath));
  const writeRoot = await realpath(absolute(config.writeRoot));
  const writer = await lstat(writeRoot);
  if (!writer.isDirectory() || (writer.mode & 0o077) !== 0 || (process.getuid && writer.uid !== process.getuid())) fail('CONFIG_PERMISSIONS', 'Write root must be an owner only directory');
  if (within(writeRoot, binaryPath) || within(writeRoot, await realpath(configPath))) fail('INVALID_CONFIG', 'Executable and configuration must be outside the output root');
  const readRoots = [];
  for (const value of config.readRoots) {
    const root = await realpath(absolute(value));
    if (!(await lstat(root)).isDirectory()) fail('INVALID_CONFIG', 'Read roots must be directories');
    readRoots.push(root);
  }
  const timeoutMs = config.timeoutMs ?? 120000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 300000) fail('INVALID_CONFIG', 'Timeout must be 100 through 300000 milliseconds');
  if (await fileDigest(binaryPath, 256 * 1024 * 1024) !== config.binarySha256) fail('BINARY_DRIFT', 'Native executable differs from the operator pin');
  return { binaryPath, binarySha256: config.binarySha256, writeRoot, readRoots: [...readRoots, writeRoot], policySha256s: config.policySha256s, timeoutMs };
}
async function readPath(config, value, limit = 64 * 1024 * 1024 * 1024) {
  const path = await realpath(absolute(value));
  if (!config.readRoots.some((root) => within(root, path))) fail('PATH_DENIED', 'Input is outside configured read roots');
  const { handle } = await fileHandle(path, limit); await handle.close();
  return path;
}
async function outputPath(config, value) {
  const requested = absolute(value); const parent = await realpath(dirname(requested));
  if (!within(config.writeRoot, parent) || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(basename(requested))) fail('PATH_DENIED', 'Output must be a new file inside the configured write root');
  const path = resolve(parent, basename(requested));
  try { await lstat(path); fail('OUTPUT_EXISTS', 'Output exists; reconcile prior results before retrying'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  return path;
}
export async function buildRequest(name, unsafe, config) {
  const def = definitions.find(([tool]) => tool === name);
  if (!def) fail('UNKNOWN_TOOL', 'Unknown tool');
  const args = object(unsafe); exact(args, def[2], def[3]);
  for (const value of Object.values(args)) absolute(value);
  const op = name.slice('weight_'.length);
  const native = { op }; let receiptOutput;
  for (const field of ['source', 'patch', 'policy', 'receipt', 'evaluation', 'artifact']) if (args[field]) native[field] = await readPath(config, args[field], field === 'patch' ? 128 * 1024 * 1024 : ['policy', 'receipt', 'evaluation'].includes(field) ? 1048576 : undefined);
  if (args.policy && !config.policySha256s.includes(await fileDigest(native.policy, 1048576))) fail('POLICY_DENIED', 'Policy bytes are not approved by the operator configuration');
  if (args.governanceEvidence) native.governance_evidence = await readPath(config, args.governanceEvidence, 1048576);
  if (args.output) native.output = await outputPath(config, args.output);
  if (args.receiptOutput) {
    receiptOutput = await outputPath(config, args.receiptOutput);
    if (receiptOutput === native.output) fail('INVALID_PATH', 'Checkpoint and receipt output paths must differ');
  }
  if (args.tensorList) {
    const list = JSON.parse((await bytes(await readPath(config, args.tensorList, 65536), 65536)).toString('utf8'));
    if (!Array.isArray(list) || list.length < 1 || list.length > 256 || new Set(list).size !== list.length || list.some((v) => typeof v !== 'string' || !v.length || v.length > 512 || /[\x00-\x1f\x7f]/u.test(v))) fail('INVALID_INPUT', 'Tensor selection requires 1 through 256 unique bounded names');
    native.tensors = list;
  }
  if (args.expectedDigestFile) {
    const value = JSON.parse((await bytes(await readPath(config, args.expectedDigestFile, 1024), 1024)).toString('utf8'));
    exact(value, ['sha256']);
    if (typeof value.sha256 !== 'string' || !HASH.test(value.sha256)) fail('INVALID_INPUT', 'Expected artifact digest is invalid');
    native.expected_sha256 = value.sha256;
  }
  return { native, receiptOutput };
}

export async function executeTool(name, args, options = {}) {
  const config = await loadConfig(options.configPath);
  const { native, receiptOutput } = await buildRequest(name, args, config);
  const request = JSON.stringify(native);
  if (Buffer.byteLength(request) > MAX_REQUEST) fail('RESOURCE_LIMIT', 'Native request exceeds bound');
  if (options.signal?.aborted) fail('CANCELLED', 'Native operation cancelled before launch');
  const result = await new Promise((resolveResult, reject) => {
    const child = spawn(config.binaryPath, [], { cwd: config.writeRoot, env: { PATH: '/usr/bin:/bin', LANG: 'C', TZ: 'UTC', TMPDIR: config.writeRoot }, stdio: ['pipe', 'pipe', 'pipe'], shell: false });
    options.child?.(child);
    let stdout = Buffer.alloc(0); let stderr = Buffer.alloc(0); let terminated;
    const stop = (code, message) => { if (!terminated) { terminated = new BoundaryError(code, message); child.kill('SIGKILL'); } };
    const timer = setTimeout(() => stop('TIMEOUT', 'Native operation timed out; reconcile output before retrying'), config.timeoutMs);
    const abort = () => stop('CANCELLED', 'Native operation cancelled; reconcile output before retrying');
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    child.stdout.on('data', (chunk) => { if (stdout.length + chunk.length > MAX_RESULT) stop('RESOURCE_LIMIT', 'Native output exceeds bound'); else stdout = Buffer.concat([stdout, chunk]); });
    child.stderr.on('data', (chunk) => { if (stderr.length + chunk.length > 65536) stop('RESOURCE_LIMIT', 'Native diagnostic exceeds bound'); else stderr = Buffer.concat([stderr, chunk]); });
    child.on('error', () => { terminated ??= new BoundaryError('NATIVE_UNAVAILABLE', 'Pinned native runtime could not start'); });
    child.stdin.on('error', () => {});
    child.on('close', (code) => {
      clearTimeout(timer); options.signal?.removeEventListener('abort', abort);
      if (terminated) return reject(terminated);
      if (code !== 0) return reject(new BoundaryError('NATIVE_REJECTED', 'Native operation refused; inspect the bounded inputs and reconcile output before retrying'));
      try { const value = object(JSON.parse(stdout.toString('utf8'))); if (value.activation_authorized === true) fail('INVALID_NATIVE_RESULT', 'Runtime cannot grant activation authority'); resolveResult(value); }
      catch (error) { reject(error); }
    });
    child.stdin.end(request);
  });
  if (receiptOutput) {
    try { await writeFile(receiptOutput, `${JSON.stringify(result, null, 2)}\n`, { flag: 'wx', mode: 0o600 }); }
    catch { fail('RECEIPT_WRITE_FAILED', 'Checkpoint may have completed but receipt persistence failed; reconcile output before retrying'); }
  }
  return { runtimeSha256: config.binarySha256, activationAuthorized: false, result, ...(receiptOutput ? { receiptOutput } : {}) };
}

export function startServer() {
  let pending = Buffer.alloc(0); let initialized = false; let active;
  const send = (id, result, error) => process.stdout.write(`${JSON.stringify(error ? { jsonrpc: '2.0', id, error } : { jsonrpc: '2.0', id, result })}\n`);
  const toolError = (error) => ({ isError: true, content: [{ type: 'text', text: JSON.stringify({ code: error.code ?? 'INVALID_INPUT', message: error instanceof BoundaryError ? error.message : 'Request failed validation', activationAuthorized: false }) }] });
  async function handle(value) {
    let id = null;
    try {
      const request = object(value);
      if (request.jsonrpc !== '2.0' || typeof request.method !== 'string') fail('INVALID_INPUT', 'Invalid JSON RPC request');
      if (!Object.hasOwn(request, 'id')) {
        if (request.method === 'notifications/cancelled' && active?.id === request.params?.requestId) active.controller.abort();
        return;
      }
      id = request.id;
      if (!(typeof id === 'string' && id.length <= 128) && !(typeof id === 'number' && Number.isSafeInteger(id))) fail('INVALID_INPUT', 'Invalid request identifier');
      if (request.method === 'initialize') { initialized = true; send(id, { protocolVersion: ['2024-11-05', '2025-03-26', '2025-06-18'].includes(request.params?.protocolVersion) ? request.params.protocolVersion : '2025-06-18', capabilities: { tools: { listChanged: false } }, serverInfo: { name: NAME, version: VERSION }, instructions: 'Native research artifacts only. No model activation authority. Use operator approved paths and policy bytes.' }); return; }
      if (!initialized) fail('NOT_INITIALIZED', 'Initialize the MCP session first');
      if (request.method === 'ping') { send(id, {}); return; }
      if (request.method === 'tools/list') { send(id, { tools: TOOLS }); return; }
      if (request.method !== 'tools/call') { send(id, null, { code: -32601, message: 'Method not found' }); return; }
      const params = object(request.params); exact(params, ['name', 'arguments'], ['_meta']);
      // MCP Request.params allows opaque client metadata beside tool arguments.
      // The full wire request is capped at 64 KiB. Never merge metadata into the
      // argument object or native request, and never interpret it as authority.
      if (Object.hasOwn(params, '_meta')) {
        const metadata = object(params._meta);
        if (Object.hasOwn(metadata, 'progressToken') && typeof metadata.progressToken !== 'string' &&
          !(typeof metadata.progressToken === 'number' && Number.isFinite(metadata.progressToken))) fail('INVALID_INPUT', 'Invalid progress token');
      }
      if (active) { send(id, toolError(new BoundaryError('BUSY', 'One native operation is already running'))); return; }
      const controller = new AbortController(); active = { id, controller };
      try { const result = await executeTool(params.name, params.arguments, { signal: controller.signal }); send(id, { content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result, isError: false }); }
      catch (error) { send(id, toolError(error)); }
      finally { active = undefined; }
    } catch { send(id, null, { code: -32600, message: 'Invalid request' }); }
  }
  process.stdin.on('data', (chunk) => {
    pending = Buffer.concat([pending, chunk]);
    while (true) {
      const newline = pending.indexOf(10);
      if (newline < 0) break;
      const line = pending.subarray(0, newline); pending = pending.subarray(newline + 1);
      if (line.length > MAX_REQUEST) { send(null, null, { code: -32700, message: 'Request exceeds bound' }); continue; }
      if (!line.length) continue;
      try { void handle(JSON.parse(line.toString('utf8'))); }
      catch { send(null, null, { code: -32700, message: 'Invalid JSON' }); }
    }
    if (pending.length > MAX_REQUEST) { active?.controller.abort(); send(null, null, { code: -32700, message: 'Request exceeds bound' }); process.stdin.destroy(); pending = Buffer.alloc(0); }
  });
  process.on('SIGTERM', () => { active?.controller.abort(); process.stdin.destroy(); });
  process.on('SIGINT', () => { active?.controller.abort(); process.stdin.destroy(); });
  process.stdin.on('end', () => { active?.controller.abort(); });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) startServer();
