// Copy the files that are not markdown into the site's public folder, so
// that the grammar travels with the documentation.
//
// The grammar lives in the project's `docs/spec/grammar.ebnf`; this keeps
// the site from carrying a second copy that could drift.

import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = join(here, '..');
const repo = join(site, '..');

const assets = [['docs/spec/grammar.ebnf', 'grammar.ebnf']];

for (const [from, to] of assets) {
  const target = join(site, 'public', to);
  await mkdir(dirname(target), { recursive: true });
  await copyFile(join(repo, from), target);
  console.log(`copied ${from} -> public/${to}`);
}
