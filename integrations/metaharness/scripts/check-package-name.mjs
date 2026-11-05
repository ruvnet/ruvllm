#!/usr/bin/env node
// Release guard: this package must never be published as @ruvector/ruvllm.
// That name belongs to the inference runtime built from ruvnet/RuVector
// (npm/packages/ruvllm). Publishing this harness under it (4.0.0, 4.0.1)
// replaced the runtime for every `npm i @ruvector/ruvllm` user.
import { readFile } from 'node:fs/promises';

const RESERVED = new Set(['@ruvector/ruvllm', 'ruvllm']);
const EXPECTED = '@ruvector/ruvllm-gpu-loop';
const manifest = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

if (RESERVED.has(manifest.name)) {
  console.error(`Refusing to publish: package name "${manifest.name}" is reserved for the RuVector inference runtime. Use "${EXPECTED}".`);
  process.exit(1);
}
if (manifest.name !== EXPECTED) {
  console.error(`Refusing to publish: expected package name "${EXPECTED}", found "${manifest.name}".`);
  process.exit(1);
}
if (manifest.bin && Object.hasOwn(manifest.bin, 'ruvllm')) {
  console.error('Refusing to publish: the "ruvllm" bin belongs to @ruvector/ruvllm; this package ships "ruvllm-gpu".');
  process.exit(1);
}
console.log(`package name guard: ${manifest.name}@${manifest.version} ok`);
