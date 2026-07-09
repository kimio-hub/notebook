import { defineCollection, z } from "astro:content"
import { glob } from "astro/loaders"

const notes = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/notes" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    tags: z.array(z.string()).default([]),
    category: z.enum(["basics", "papers", "other"]).default("other"),
    date: z.coerce.date().optional(),
    updated: z.coerce.date().optional(),
    draft: z.boolean().default(false),
  }),
})

export const collections = { notes }
