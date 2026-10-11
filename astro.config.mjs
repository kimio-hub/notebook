import { defineConfig } from "astro/config"
import sitemap from "@astrojs/sitemap"
import { unified } from "@astrojs/markdown-remark"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"

// https://astro.build/config
// Cloudflare Pages: build command `npm run build`, output directory `dist`
export default defineConfig({
  site: "https://hides.cc.cd",
  output: "static",
  outDir: "dist",
  integrations: [sitemap()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
    // Mermaid blocks stay as plain code and are rendered client-side (see BaseLayout).
    syntaxHighlight: { type: "shiki", excludeLangs: ["math", "mermaid"] },
    shikiConfig: {
      themes: { light: "github-light", dark: "github-dark-dimmed" },
      wrap: true,
    },
  },
  vite: {
    server: {
      watch: {
        ignored: ["**/dist/**", "**/node_modules/**"],
      },
    },
  },
})
