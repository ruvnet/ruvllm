#!/usr/bin/env node
import { runExperiment } from './engine.mjs';
import { createFixture } from './fixture.mjs';

const [command, ...args] = process.argv.slice(2);
function options(allowed) {
  const result = {};
  for (let i = 0; i < args.length; i += 2) {
    if (!allowed.includes(args[i]) || !args[i + 1] || Object.hasOwn(result, args[i])) throw new Error('Unknown, duplicate or missing argument');
    result[args[i]] = args[i + 1];
  }
  return result;
}
try {
  let result;
  if (command === 'run') {
    const flags = options(['--config', '--plan']);
    result = await runExperiment({ configPath: flags['--config'], planPath: flags['--plan'] });
  } else if (command === 'fixture') {
    const flags = options(['--dir', '--binary']);
    if (!flags['--dir'] || !flags['--binary']) throw new Error('fixture requires --dir and --binary');
    result = await runExperiment(await createFixture({ directory: flags['--dir'], binaryPath: flags['--binary'] }));
  } else if (command === '--help' || command === 'help') {
    console.log('Usage: ruvllm-experiment run --config ABS --plan ABS\n       ruvllm-experiment fixture --dir NEW_ABS --binary ABS');
    process.exit(0);
  } else throw new Error('Expected run, fixture or --help');
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(JSON.stringify({ ok: false, error: error.message, productionActivation: false })); process.exitCode = 1;
}
