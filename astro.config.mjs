import { defineConfig } from "astro/config"
import sitemap from "@astrojs/sitemap"
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
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: "github-dark-dimmed",
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
