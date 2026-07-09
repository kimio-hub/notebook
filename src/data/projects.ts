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
  /** Main external or demo link */
  href: string
  links?: ProjectLink[]
}

/**
 * 已完成 / 进行中的项目列表。
 * 新增项目：在本数组末尾追加一项即可。
 */
export const projects: Project[] = [
  {
    title: "项目复盘 · 知乎专栏",
    description:
      "已完成项目的公开记录与总结。完整内容发布在知乎专栏，点击卡片可阅读原文。",
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
