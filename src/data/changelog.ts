export type ChangelogItem = {
  date: string // YYYY-MM-DD
  title: string
  summary: string
  type: "site" | "note" | "project" | "life"
  href?: string
}

/**
 * 站点里程碑 / 可视化更新流
 * 新笔记会由内容库自动并入 Updates 页；这里放「站点本身」的变更。
 */
export const changelog: ChangelogItem[] = [
  {
    date: "2026-07-10",
    title: "新增项目页与知乎入口",
    summary: "上线项目作品集页面，接入知乎主页链接；首个公开项目为知乎专栏复盘。",
    type: "project",
    href: "/projects/",
  },
  {
    date: "2026-07-10",
    title: "个人站 2.0 上线",
    summary: "从笔记花园重构为完整个人网站：首页动态、笔记卡片、友链页、可视化更新时间线。移除 RSS。",
    type: "site",
    href: "/",
  },
  {
    date: "2026-07-10",
    title: "笔记结构重整",
    summary: "公开笔记分为「基础」与「论文精读」两条线，同步脚本与目录页一并升级。",
    type: "note",
    href: "/notes/",
  },
  {
    date: "2026-07-09",
    title: "站点主题与首页改版",
    summary: "统一视觉语言，强化主页导航与阅读入口。",
    type: "site",
  },
]

export const typeLabel: Record<ChangelogItem["type"], string> = {
  site: "站点",
  note: "笔记",
  project: "项目",
  life: "生活",
}
