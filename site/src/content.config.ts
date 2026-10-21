import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

/**
 * The reference documentation.
 *
 * This is the project's own `docs/` folder, read in place: the spec, the
 * architecture notes and the roadmap are written once.
 */
const reference = defineCollection({
  loader: glob({ pattern: '**/*.md', base: '../docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    section: z.string().default('Reference'),
    order: z.number().default(99),
  }),
});

/** The guides: the things a person reads in order. */
const guides = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/guides' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    section: z.string().default('Guides'),
    order: z.number().default(99),
  }),
});

export const collections = { reference, guides };
