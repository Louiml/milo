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
    if (!node.url) return;

    if (node.url.startsWith('/')) {
      // A link to a route on the site. It needs the base, because the
      // site is published under a path.
      node.url = `${base}${node.url.replace(/^\//, '')}`;
      return;
    }

    if (!node.url.startsWith('.')) return;

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
      node.url = `${base}${basename(target)}`;
      return;
    }

    const name = parts.pop() ?? '';
    if (!name.endsWith('.md')) {
      // The grammar, and anything else that is not a page.
      node.url = `${base}${name}`;
      return;
    }

    // The slug has to match the one Astro gives the page, or the link
    // points at a route that does not exist: lower case, with the
    // punctuation turned into dashes.
    const slug = name
      .replace(/\.md$/, '')
      .toLowerCase()
      .replace(/[^\w\s-]/g, '-')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');

    parts.push(slug);
    node.url = `${base}${collection}/${parts.join('/')}/`;
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

/**
 * Put the base on every link that points at the site.
 *
 * The site is published under a path, so a link to `/docs/roadmap/` has to
 * become `/milo/docs/roadmap/`. This runs as a rehype plugin, after Astro
 * has turned the relative links in the documentation into routes, so it
 * sees the final URLs and can put the base on all of them.
 */
function rehypeMiloBase() {
  return (tree) => {
    const walk = (node) => {
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      if (!node || typeof node !== 'object') return;
      if (node.type === 'element' && node.tagName === 'a') {
        const href = node.properties && node.properties.href;
        if (typeof href === 'string' && href.startsWith('/') && !href.startsWith(base)) {
          node.properties.href = base + href.replace(/^\//, '');
        }
      }
      if (node.children) walk(node.children);
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
//
// The site is published on GitHub Pages, under the repository's own
// path, so every link has to carry the base.

/** The path the site is published under. */
const base = '/milo/';

export default defineConfig({
  site: 'https://louiml.github.io/milo',
  base,
  trailingSlash: 'ignore',
  build: {
    format: 'directory',
  },
  markdown: {
    remarkPlugins: [remarkMiloLinks],
    rehypePlugins: [rehypeMiloBase],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
