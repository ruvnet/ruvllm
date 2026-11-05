// Minimal ustar writer for tests: entries are { name, type = '0', body = '', linkname = '' }.
// `longName: true` emits a GNU 'L' long-name record before the entry.
function header({ name, type = '0', size = 0, linkname = '', magic = 'ustar\0', version = '00' }) {
  const block = Buffer.alloc(512);
  block.write(name.slice(0, 100), 0, 'utf8');
  block.write('0000644\0', 100); block.write('0001750\0', 108); block.write('0001750\0', 116);
  block.write(size.toString(8).padStart(11, '0') + '\0', 124);
  block.write('00000000000\0', 136);
  block.write('        ', 148);
  block.write(type, 156);
  block.write(linkname.slice(0, 100), 157);
  block.write(magic, 257, 'latin1'); block.write(version, 263, 'latin1');
  let sum = 0; for (const byte of block) sum += byte;
  block.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
  return block;
}
const pad = bytes => Buffer.concat([bytes, Buffer.alloc((512 - (bytes.length % 512)) % 512)]);

export function tarArchive(entries) {
  const blocks = [];
  for (const entry of entries) {
    const body = Buffer.from(entry.body ?? '');
    if (entry.longName) {
      const name = Buffer.from(entry.name + '\0');
      blocks.push(header({ name: '././@LongLink', type: 'L', size: name.length, magic: 'ustar  \0', version: '' }), pad(name));
      blocks.push(header({ ...entry, name: entry.name.slice(0, 100), size: body.length, magic: 'ustar  \0', version: '' }), pad(body));
    } else {
      blocks.push(header({ ...entry, size: body.length }), pad(body));
    }
  }
  blocks.push(Buffer.alloc(1024));
  return Buffer.concat(blocks);
}
