import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"
import { localizePath } from "../i18n/utils"

export type ProjectLink = {
  label: string
  labelEn?: string
  href: string
}

export type Project = {
  id: string
  title: string
  titleEn: string
  description: string
  descriptionEn: string
  date?: string
  status: "completed" | "wip" | "archived"
  tags?: string[]
  /** Unprefixed internal path or external URL */
  href: string
  links?: ProjectLink[]
}

export function isExternalHref(href: string): boolean {
  return /^https?:\/\//i.test(href)
}

export const projects: Project[] = [
  {
    id: "gsm8k-rl-5090",
    title: "单卡 5090 上的 GSM8K 数学推理 RL",
    titleEn: "GSM8K Math Reasoning RL on a Single RTX 5090",
    description:
      "Qwen2.5-7B-Instruct + QLoRA + GRPO，GSM8K test[:500] 从 0.796 做到 0.866。完整记录 reward、RFT、dynamic advantage、Muon、pass@32 与全对组 anchor 的消融。",
    descriptionEn:
      "Qwen2.5-7B-Instruct + QLoRA + GRPO on GSM8K test[:500]: 0.796 → 0.866. Full write-up of reward design, RFT, dynamic advantage, Muon, pass@32, and all-correct-group anchors.",
    date: "2026-07",
    status: "completed",
    tags: ["RL", "GRPO", "GSM8K", "Qwen", "5090"],
    href: "/projects/gsm8k-rl-5090/",
    links: [
      {
        label: "知乎专栏（相关）",
        labelEn: "Zhihu column",
        href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
      },
    ],
  },
  {
    id: "zhihu-column",
    title: "项目复盘 · 知乎专栏",
    titleEn: "Project write-up · Zhihu",
    description: "已完成项目的公开记录与总结，完整内容见知乎专栏原文。",
    descriptionEn: "Public project notes and retrospectives on Zhihu.",
    date: "2026",
    status: "completed",
    tags: ["项目", "复盘", "知乎"],
    href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
    links: [
      {
        label: "知乎专栏",
        labelEn: "Zhihu column",
        href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
      },
    ],
  },
]

export function projectTitle(p: Project, lang: Locale) {
  return lang === "en" ? p.titleEn : p.title
}

export function projectDescription(p: Project, lang: Locale) {
  return lang === "en" ? p.descriptionEn : p.description
}

export function projectHref(p: Project, lang: Locale) {
  return isExternalHref(p.href) ? p.href : localizePath(p.href, lang)
}

export function projectLinkLabel(link: ProjectLink, lang: Locale) {
  return lang === "en" && link.labelEn ? link.labelEn : link.label
}

export function statusLabel(status: Project["status"], lang: Locale) {
  if (status === "completed") return t(lang, "projects.statusCompleted")
  if (status === "wip") return t(lang, "projects.statusWip")
  return t(lang, "projects.statusArchived")
}
