// Optional host handoff: writes job requests, never contacts a GPU provider or resolves secrets.
import { constants } from 'node:fs';
import { open, lstat, realpath, writeFile } from 'node:fs/promises';
import { join, resolve, isAbsolute } from 'node:path';
import { parseJson } from './engine.mjs';

const ensure = (test, message) => { if (!test) throw new Error(message); };
const safeId = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(value);

async function spoolRoot() {
  const root = process.env.RUVLLM_GPU_SPOOL_ROOT;
  ensure(typeof root === 'string' && isAbsolute(root) && resolve(root) === root, 'SPOOL_NOT_CONFIGURED');
  ensure(await realpath(root) === root, 'SPOOL_PATH_NOT_CANONICAL');
  const stat = await lstat(root);
  ensure(stat.isDirectory() && !stat.isSymbolicLink() && (stat.mode & 0o077) === 0
    && stat.uid === process.getuid(), 'SPOOL_NOT_PRIVATE');
  return root;
}

async function writeOnce(path, value) {
  const text = JSON.stringify(value, null, 2) + '\n';
  ensure(Buffer.byteLength(text) <= 65536, 'SPOOL_REQUEST_TOO_LARGE');
  try { await writeFile(path, text, { flag: 'wx', mode: 0o600 }); }
  catch (error) {
    if (error.code !== 'EEXIST') throw new Error('SPOOL_WRITE_FAILED');
    const prior = await readReply(path);
    ensure(JSON.stringify(prior) === JSON.stringify(value), 'SPOOL_IDENTITY_CONFLICT');
  }
}

async function readReply(path) {
  let handle;
  try { handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK); }
  catch (error) { if (error.code === 'ENOENT') return null; throw new Error('SPOOL_READ_FAILED'); }
  try {
    const stat = await handle.stat();
    ensure(stat.isFile() && stat.size <= 65536 && (stat.mode & 0o077) === 0
      && stat.uid === process.getuid(), 'SPOOL_REPLY_NOT_PRIVATE');
    const bytes = Buffer.alloc(stat.size);
    const { bytesRead } = await handle.read(bytes, 0, bytes.length, 0);
    ensure(bytesRead === bytes.length, 'SPOOL_REPLY_CHANGED');
    return parseJson(bytes);
  } finally { await handle.close(); }
}

export async function submit({ job, idempotencyKey, signal }) {
  ensure(!signal?.aborted && safeId(idempotencyKey), 'SPOOL_INVALID_REQUEST');
  const root = await spoolRoot();
  await writeOnce(join(root, `${idempotencyKey}.request.json`), {
    schema: 'ruvllm-gpu-spool-request/v1', runId: idempotencyKey, job,
    credentialHandling: 'host-resolves-secretRef-no-secret-values-in-spool',
  });
  return { runId: idempotencyKey };
}

export async function inspect({ runId, signal }) {
  ensure(!signal?.aborted && safeId(runId), 'SPOOL_INVALID_RUN_ID');
  const root = await spoolRoot();
  const reply = await readReply(join(root, `${runId}.status.json`));
  if (!reply) return { status: 'running' };
  ensure(reply.schema === 'ruvllm-gpu-spool-status/v1' && reply.runId === runId
    && ['running', 'completed', 'failed'].includes(reply.status)
    && Object.keys(reply).every(key => ['schema', 'runId', 'status', 'costUsd', 'completion'].includes(key)), 'SPOOL_INVALID_STATUS');
  return { status: reply.status, ...(reply.costUsd === undefined ? {} : { costUsd: reply.costUsd }),
    ...(reply.completion === undefined ? {} : { completion: reply.completion }) };
}

export async function cleanup({ runId, job, signal }) {
  ensure(!signal?.aborted && safeId(runId), 'SPOOL_INVALID_RUN_ID');
  const root = await spoolRoot();
  await writeOnce(join(root, `${runId}.cleanup-request.json`), {
    schema: 'ruvllm-gpu-spool-cleanup-request/v1', runId, jobId: job.id,
  });
  const reply = await readReply(join(root, `${runId}.cleanup.json`));
  if (!reply) return { released: false };
  ensure(reply.schema === 'ruvllm-gpu-spool-cleanup/v1' && reply.runId === runId
    && typeof reply.released === 'boolean'
    && Object.keys(reply).every(key => ['schema', 'runId', 'released'].includes(key)), 'SPOOL_INVALID_CLEANUP');
  return { released: reply.released };
}
