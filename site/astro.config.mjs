// @ts-check
import { defineConfig } from 'astro/config';
import { basename, dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

/** The project's own documentation folder. */
const referenceBase = resolve(repoRoot, 'docs');
/** The guides, which live with the site. */
const guidesBase = resolve(repoRoot, 'site', 'src', 'content', 'guides');

/**
 * Rewrite the project's relative links into routes on the site.
 *
 * The reference documentation is read straight out of the repository's
 * `docs/` folder, so its links point at files: `../roadmap.md`,
 * `./grammar.ebnf`. On the site those have to become routes. This plugin
 * does that, so the documentation is written once and works in both places.
 */
function remarkMiloLinks() {
  /** @param {{url: string}} node */
  const rewrite = (node, filePath) => {
    if (!node.url || !node.url.startsWith('.')) return;

    const target = resolve(dirname(filePath), node.url);

    let collection;
    let parts;
    if (target.startsWith(referenceBase)) {
      collection = 'docs';
      parts = relative(referenceBase, target).split(/[\\/]/);
    } else if (target.startsWith(guidesBase)) {
      collection = 'guides';
      parts = relative(guidesBase, target).split(/[\\/]/);
    } else {
      // Something else: serve it as a plain file.
      node.url = '/' + basename(target);
      return;
    }

    const name = parts.pop() ?? '';
    if (!name.endsWith('.md')) {
      // The grammar, and anything else that is not a page.
      node.url = '/' + name;
      return;
    }

    const slug = name
      .replace(/\.md$/, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');

    parts.push(slug);
    node.url = `/${collection}/${parts.join('/')}/`;
  };

  return (tree, file) => {
    const filePath = file?.history?.[0] ?? file?.path;
    if (!filePath) return;

    const walk = (node) => {
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      if (node && typeof node === 'object') {
        if (node.type === 'link') rewrite(node, filePath);
        for (const value of Object.values(node)) walk(value);
      }
    };
    walk(tree);
  };
}

// The documentation site for Milo.
//
// The reference documentation is not written twice: the site's content
// collections read the repository's own `docs/` folder, so the spec, the
// architecture notes and the roadmap are the same files that live in the
// project.

export default defineConfig({
  site: 'https://milo-lang.org',
  trailingSlash: 'ignore',
  build: {
    format: 'directory',
  },
  markdown: {
    remarkPlugins: [remarkMiloLinks],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
