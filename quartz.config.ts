import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Kimio's personal site — "Ink Garden"
 * Warm parchment + deep ink + copper/teal accents.
 * Editorial digital garden for RL / LLM notes.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Kimio",
    pageTitleSuffix: " · 笔记",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "zh-CN",
    baseUrl: "hides.cc.cd",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Fraunces",
        body: "Source Serif 4",
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#f6f1e7",
          lightgray: "#e5dccb",
          gray: "#9a8f7e",
          darkgray: "#3f3a34",
          dark: "#1c1915",
          secondary: "#0f6b63",
          tertiary: "#b45309",
          highlight: "rgba(15, 107, 99, 0.10)",
          textHighlight: "#f0c67488",
        },
        darkMode: {
          light: "#12151a",
          lightgray: "#2a3038",
          gray: "#7a8494",
          darkgray: "#d5d0c6",
          dark: "#f0ebe3",
          secondary: "#5ec4b6",
          tertiary: "#e0a45a",
          highlight: "rgba(94, 196, 182, 0.12)",
          textHighlight: "#b4530988",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
