#!/usr/bin/env node
import { setTimeout as wait } from 'node:timers/promises';
import { tickLoop, readLoopStatus, readLoopConfig } from './gpu-loop.mjs';
import { loadHostAdapter } from './gpu-host.mjs';

const usage = 'Usage: node src/gpu-loop-cli.mjs tick|status|run --config ABS\n'
  + '  tick/run require --adapter ABS.mjs --adapter-sha256 SHA256\n'
  + '  run accepts --max-ticks 1..10000 (default 1); no infinite loop is started.\n'
  + 'The trusted host resolves ruOS secret references and executes Vast.ai operations.\n'
  + 'No provider or secret access is supplied by this CLI.\n';

function options(argv, command) {
  const result = {};
  const allowed = command === 'status' ? ['--config']
    : ['--config', '--adapter', '--adapter-sha256', ...(command === 'run' ? ['--max-ticks'] : [])];
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index], value = argv[index + 1];
    if (!allowed.includes(key) || !value || value.startsWith('--') || Object.hasOwn(result, key)) throw new Error('INVALID_ARGUMENTS');
    result[key] = value;
  }
  if (!result['--config'] || (command !== 'status' && (!result['--adapter'] || !result['--adapter-sha256']))) throw new Error('INVALID_ARGUMENTS');
  const maxTicks = Number(result['--max-ticks'] ?? 1);
  if (!Number.isSafeInteger(maxTicks) || maxTicks < 1 || maxTicks > 10000) throw new Error('INVALID_ARGUMENTS');
  return { ...result, maxTicks };
}

const [command, ...argv] = process.argv.slice(2);
if (command === '--help' || command === 'help') process.stdout.write(usage);
else if (!['tick', 'status', 'run'].includes(command)) {
  process.stderr.write(usage); process.exitCode = 2;
} else {
  const stop = new AbortController();
  const stopLoop = () => stop.abort();
  process.once('SIGINT', stopLoop); process.once('SIGTERM', stopLoop);
  try {
    const flags = options(argv, command);
    const configPath = flags['--config'];
    if (command === 'status') process.stdout.write(JSON.stringify(await readLoopStatus({ configPath })) + '\n');
    else {
      const ticks = command === 'run' ? flags.maxTicks : 1;
      for (let index = 0; index < ticks && !stop.signal.aborted; index++) {
        // Revalidate code and immutable config before each tick. Tick locks and
        // durable submission markers own duplicate prevention across processes.
        const config = await readLoopConfig({ configPath });
        const adapter = await loadHostAdapter({ modulePath: flags['--adapter'], sha256: flags['--adapter-sha256'] });
        const state = await tickLoop({ configPath, adapter });
        process.stdout.write(JSON.stringify(state) + '\n');
        if (index + 1 < ticks && !stop.signal.aborted) {
          // The scheduler may shorten nextTickAt to a job deadline or immediate
          // cleanup. Never sleep the full cadence across those boundaries.
          const delayMs = Math.max(10, Math.min(config.intervalSeconds * 1000, state.nextTickAt - Date.now()));
          try { await wait(delayMs, undefined, { signal: stop.signal }); }
          catch (error) { if (error.name !== 'AbortError') throw error; }
        }
      }
    }
  } catch (error) {
    // Raw host exceptions can carry credentials. Surface only scheduler-owned
    // fixed codes; detailed provider diagnostics remain with the host.
    const code = typeof error.code === 'string' && /^[A-Z][A-Z0-9_]{0,63}$/.test(error.code)
      ? error.code : 'GPU_LOOP_FAILED';
    process.stderr.write(JSON.stringify({ ok: false, code, productionActivation: false }) + '\n');
    process.exitCode = 1;
  } finally {
    process.removeListener('SIGINT', stopLoop); process.removeListener('SIGTERM', stopLoop);
  }
}
