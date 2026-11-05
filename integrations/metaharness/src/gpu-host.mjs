// The operator supplies this module locally. Secret resolution and provider APIs
// remain with that host; this loader never fetches credentials or parses stdout.
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import nodeModule from 'node:module';
import { pathToFileURL } from 'node:url';
import { checkedPath } from './engine.mjs';

function failure(code) { const error = new Error(code); error.code = code; return error; }

// Import the entry module from the bytes that were hashed, not by re-reading the
// path: a loader hook answers the pinned URL with those exact bytes, so a file
// swapped after the check is never executed. Relative imports still resolve
// next to the original file (they are trusted operator code, as before).
const ASYNC_HOOKS = 'data:text/javascript,' + encodeURIComponent(`
const sources = new Map();
export async function initialize({ port }) {
  port.on('message', ({ href, source }) => { sources.set(href, source); port.postMessage(href); });
  port.unref();
}
export async function load(url, context, nextLoad) {
  const source = sources.get(url);
  if (source === undefined) return nextLoad(url, context);
  sources.delete(url);
  return { format: 'module', source, shortCircuit: true };
}`);
let asyncPort;
const pending = new Map();

function asyncHooks() {
  if (!asyncPort) {
    const { port1, port2 } = new MessageChannel();
    port1.on('message', href => { pending.get(href)?.(); pending.delete(href); });
    port1.unref();
    nodeModule.register(ASYNC_HOOKS, { data: { port: port2 }, transferList: [port2] });
    asyncPort = port1;
  }
  return asyncPort;
}

async function importVerified(href, bytes) {
  if (typeof nodeModule.registerHooks === 'function') {
    const hooks = nodeModule.registerHooks({
      load(url, context, nextLoad) {
        if (url === href) return { format: 'module', source: bytes, shortCircuit: true };
        return nextLoad(url, context);
      },
    });
    try { return await import(href); } finally { hooks.deregister(); }
  }
  // Node 22 before module.registerHooks (22.15): off-thread loader hooks.
  const port = asyncHooks();
  port.ref();
  try {
    await new Promise(resolve => { pending.set(href, resolve); port.postMessage({ href, source: bytes }); });
  } finally { port.unref(); }
  return import(href);
}

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
    module = await importVerified(url.href, bytes);
  } catch { throw failure('HOST_ADAPTER_UNAVAILABLE'); }
  if (!['submit', 'inspect', 'cleanup'].every(name => typeof module[name] === 'function')) {
    throw failure('HOST_ADAPTER_INVALID_CONTRACT');
  }
  return Object.freeze(Object.fromEntries(['submit', 'inspect', 'cleanup'].map(name => [name, async input => {
    try { return await module[name](input); }
    catch { throw failure('HOST_ADAPTER_OPERATION_FAILED'); }
  }])));
}
