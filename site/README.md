# The Milo documentation site

An [Astro](https://astro.build) site. It is a static site: no server, no
database, nothing to run.

```bash
npm install
npm run dev      # a live server on http://localhost:4321
npm run build    # the static site in dist/
npm run preview  # serve the built site
```

## One source of truth

The reference documentation is **not** written twice. The site's content
collections read the project's own `docs/` folder, so the specification,
the architecture notes and the roadmap are the same files that live in the
repository:

- `docs/spec/*.md` → the reference section
- `docs/architecture/*.md` → the reference section
- `docs/roadmap.md` → the reference section
- `src/content/guides/*.md` → the guides, which are written for the site

The project's markdown links to files (`../roadmap.md`), which do not work
on a website. A small remark plugin in `astro.config.mjs` rewrites them
into routes, so the same text works in both places.

## The layout

```
src/
├── components/Sidebar.astro   the navigation
├── content/guides/             the guides
├── layouts/                    the page shells
├── pages/                      the routes
└── styles/global.css           the design system
```
