// The operator supplies this module locally. Secret resolution and provider APIs
// remain with that host; this loader never fetches credentials or parses stdout.
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { checkedPath } from './engine.mjs';

function failure(code) { const error = new Error(code); error.code = code; return error; }

export async function loadHostAdapter({ modulePath, sha256 }) {
  if (typeof sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(sha256)) throw failure('HOST_ADAPTER_INVALID_PIN');
  let path, bytes;
  try {
    path = await checkedPath(modulePath);
    if (!path.endsWith('.mjs')) throw failure('HOST_ADAPTER_INVALID_PATH');
    const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const stat = await file.stat();
      if (!stat.isFile() || stat.size > 1024 * 1024 || (stat.mode & 0o077) !== 0
        || typeof process.getuid !== 'function' || stat.uid !== process.getuid()) {
        throw failure('HOST_ADAPTER_UNSAFE_FILE');
      }
      bytes = Buffer.alloc(stat.size);
      const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
      if (bytesRead !== bytes.length) throw failure('HOST_ADAPTER_CHANGED');
    } finally { await file.close(); }
  } catch (error) {
    if (error.code?.startsWith('HOST_ADAPTER_')) throw error;
    throw failure('HOST_ADAPTER_INVALID_PATH');
  }
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) throw failure('HOST_ADAPTER_PIN_MISMATCH');
  let module;
  try {
    // The entry point and its transitive imports are trusted operator code. The
    // digest is a cache identity as well as a pin; never accept tool-supplied code.
    const url = pathToFileURL(path); url.searchParams.set('sha256', sha256);
    module = await import(url.href);
  } catch { throw failure('HOST_ADAPTER_UNAVAILABLE'); }
  if (!['submit', 'inspect', 'cleanup'].every(name => typeof module[name] === 'function')) {
    throw failure('HOST_ADAPTER_INVALID_CONTRACT');
  }
  return Object.freeze(Object.fromEntries(['submit', 'inspect', 'cleanup'].map(name => [name, async input => {
    try { return await module[name](input); }
    catch { throw failure('HOST_ADAPTER_OPERATION_FAILED'); }
  }])));
}
