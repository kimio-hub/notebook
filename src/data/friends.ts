import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"

export type Friend = {
  name: string
  url: string
  description: string
  descriptionEn?: string
  avatar?: string
  tags?: string[]
  tagsEn?: string[]
}

export const friends: Friend[] = [
  {
    name: "Fried's Blog",
    url: "https://www.friedcali.top/",
    description: "旅行、项目与生活记录 · 东方er",
    descriptionEn: "Travel, projects, and daily notes · Touhou fan",
    tags: ["博客", "项目"],
    tagsEn: ["Blog", "Projects"],
  },
  {
    name: "soyo Ling 小站",
    url: "https://www.soyoling.top/",
    description: "杂记拾遗 · 文艺作家",
    descriptionEn: "Scattered notes · literary writing",
    tags: ["博客"],
    tagsEn: ["Blog"],
  },
  {
    name: "Skywalkjian",
    url: "https://skywalkjian-site.vercel.app/",
    description: "个人站点 · 未来大佬",
    descriptionEn: "Personal site",
    tags: ["博客"],
    tagsEn: ["Blog"],
  },
]

export function friendDescription(f: Friend, lang: Locale) {
  return lang === "en" && f.descriptionEn ? f.descriptionEn : f.description
}

export function friendTags(f: Friend, lang: Locale) {
  return lang === "en" && f.tagsEn ? f.tagsEn : f.tags
}

export function friendGuide(lang: Locale) {
  return {
    title: t(lang, "friends.apply"),
    intro: lang === "en" ? "Happy to exchange links." : "欢迎互换友链。",
    requirements:
      lang === "en"
        ? [
            "Site should be stable and mostly original content.",
            "Please send site name, URL, short bio, and avatar URL.",
          ]
        : [
            "站点可以稳定访问，并以原创内容为主。",
            "请提供站点名称、链接、简介和头像地址。",
          ],
    mine: {
      name: "Kimio",
      url: "https://hides.cc.cd",
      description:
        lang === "en"
          ? "CS · RL / LLM notes and personal site"
          : "计算机 · RL / LLM 学习笔记与个人主页",
      avatar: "https://hides.cc.cd/avatar.webp",
    },
    contact:
      lang === "en"
        ? "Reach me via email or GitHub~"
        : "通过邮箱或 GitHub 联系我~",
  }
}
