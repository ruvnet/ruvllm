#!/usr/bin/env node
import { exportArchive, trainingPlan, selectDevelopmentCandidate } from './adapters.mjs';

const operations = { export: exportArchive, 'training-plan': trainingPlan, select: selectDevelopmentCandidate };
if (process.argv.length !== 3 || !Object.hasOwn(operations, process.argv[2])) {
  process.stderr.write('Usage: node src/adapters-cli.mjs export|training-plan|select < input.json\nJSON data only; no training, inference or activation is executed.\n');
  process.exitCode = process.argv[2] === '--help' ? 0 : 2;
} else {
  try {
    const chunks = [];
    let bytes = 0;
    for await (const chunk of process.stdin) {
      bytes += chunk.length;
      if (bytes > 16 * 1024 * 1024) throw new Error('Input exceeds 16 MiB');
      chunks.push(chunk);
    }
    const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    process.stdout.write(JSON.stringify(await operations[process.argv[2]](input), null, 2) + '\n');
  } catch (error) {
    process.stderr.write(JSON.stringify({ ok: false, error: error.message }) + '\n');
    process.exitCode = 1;
  }
}
