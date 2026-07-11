import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"
import { localizePath } from "../i18n/utils"

export type ChangelogItem = {
  date: string
  title: string
  titleEn: string
  summary: string
  summaryEn: string
  type: "site" | "note" | "project" | "life"
  /** Unprefixed path or external URL */
  href?: string
}

export const changelog: ChangelogItem[] = [
  {
    date: "2026-07-10",
    title: "上线中英文切换",
    titleEn: "Chinese / English language switch",
    summary: "全站界面支持中英文切换：默认中文，英文前缀 /en/，顶栏可一键切换。",
    summaryEn: "Site UI supports ZH/EN switching: Chinese by default, English under /en/, toggle in the header.",
    type: "site",
    href: "/en/",
  },
  {
    date: "2026-07-10",
    title: "发布 GSM8K 单卡 RL 实验报告",
    titleEn: "Published GSM8K single-GPU RL report",
    summary: "Qwen2.5-7B + GRPO 在 5090 上从 0.796 做到 0.866 的完整实验记录，已挂到项目页。",
    summaryEn: "Full Qwen2.5-7B + GRPO report on a 5090: 0.796 → 0.866, now on Projects.",
    type: "project",
    href: "/projects/gsm8k-rl-5090/",
  },
  {
    date: "2026-07-10",
    title: "新增项目页与知乎入口",
    titleEn: "Projects page and Zhihu link",
    summary: "上线项目作品集页面，接入知乎主页链接；首个公开项目为知乎专栏复盘。",
    summaryEn: "Launched the projects portfolio and Zhihu profile link.",
    type: "project",
    href: "/projects/",
  },
  {
    date: "2026-07-10",
    title: "个人站 2.0 上线",
    titleEn: "Personal site 2.0",
    summary: "从笔记花园重构为完整个人网站：首页动态、笔记卡片、友链页、可视化更新时间线。移除 RSS。",
    summaryEn: "Rebuilt as a full personal site: home feed, note cards, friends, timeline. RSS removed.",
    type: "site",
    href: "/",
  },
  {
    date: "2026-07-10",
    title: "笔记结构重整",
    titleEn: "Notes reorganized",
    summary: "公开笔记分为「基础」与「论文精读」两条线，同步脚本与目录页一并升级。",
    summaryEn: "Public notes split into Basics and Paper notes, with sync and folder pages updated.",
    type: "note",
    href: "/notes/",
  },
  {
    date: "2026-07-09",
    title: "站点主题与首页改版",
    titleEn: "Theme and home redesign",
    summary: "统一视觉语言，强化主页导航与阅读入口。",
    summaryEn: "Unified visual language; stronger home navigation and reading entry points.",
    type: "site",
  },
]

export function changelogTitle(item: ChangelogItem, lang: Locale) {
  return lang === "en" ? item.titleEn : item.title
}

export function changelogSummary(item: ChangelogItem, lang: Locale) {
  return lang === "en" ? item.summaryEn : item.summary
}

export function changelogHref(item: ChangelogItem, lang: Locale) {
  if (!item.href) return undefined
  if (item.href.startsWith("http")) return item.href
  return localizePath(item.href, lang)
}

export function typeLabel(type: ChangelogItem["type"], lang: Locale) {
  return t(lang, `type.${type}`)
}
