export type ProjectLink = {
  label: string
  href: string
}

export type Project = {
  title: string
  description: string
  /** YYYY-MM or YYYY-MM-DD */
  date?: string
  status: "completed" | "wip" | "archived"
  tags?: string[]
  /** Main link: internal path or external URL */
  href: string
  links?: ProjectLink[]
}

export function isExternalHref(href: string): boolean {
  return /^https?:\/\//i.test(href)
}

/**
 * 已完成 / 进行中的项目列表。
 * 站内长文放在 src/content/writeups/，href 指向 /projects/<id>/
 */
export const projects: Project[] = [
  {
    title: "单卡 5090 上的 GSM8K 数学推理 RL",
    description:
      "Qwen2.5-7B-Instruct + QLoRA + GRPO，GSM8K test[:500] 从 0.796 做到 0.866。完整记录 reward、RFT、dynamic advantage、Muon、pass@32 与全对组 anchor 的消融。",
    date: "2026-07",
    status: "completed",
    tags: ["RL", "GRPO", "GSM8K", "Qwen", "5090"],
    href: "/projects/gsm8k-rl-5090/",
    links: [
      {
        label: "知乎专栏（相关）",
        href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
      },
    ],
  },
  {
    title: "项目复盘 · 知乎专栏",
    description: "已完成项目的公开记录与总结，完整内容见知乎专栏原文。",
    date: "2026",
    status: "completed",
    tags: ["项目", "复盘", "知乎"],
    href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
    links: [
      {
        label: "知乎专栏",
        href: "https://zhuanlan.zhihu.com/p/2048752620932798368",
      },
    ],
  },
]

export const statusLabel: Record<Project["status"], string> = {
  completed: "已完成",
  wip: "进行中",
  archived: "归档",
}
