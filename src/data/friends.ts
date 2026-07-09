export type Friend = {
  name: string
  url: string
  description: string
  avatar?: string
  tags?: string[]
}

/**
 * 友链列表 — 按需编辑本文件后重新构建即可。
 * 当前同步自 Fried 站「友情链接」并保留 Fried 本人。
 */
export const friends: Friend[] = [
  {
    name: "Fried's Blog",
    url: "https://www.friedcali.top/",
    description: "旅行、项目与生活记录 · 东方er",
    tags: ["博客", "项目  "],
  },
  {
    name: "soyo Ling 小站",
    url: "https://www.soyoling.top/",
    description: "杂记拾遗 · 文艺作家",
    tags: ["博客"],
  },
  {
    name: "Skywalkjian",
    url: "https://skywalkjian-site.vercel.app/",
    description: "个人站点 · 未来大佬",
    tags: ["博客"],
  },
]

export const friendLinkGuide = {
  title: "申请友链",
  intro: "欢迎互换友链。",
  mine: {
    name: "Kimio",
    url: "https://hides.cc.cd",
    description: "计算机 · RL / LLM 学习笔记与个人主页",
    avatar: "https://hides.cc.cd/avatar.jpg",
  },
  contact: "通过邮箱或 GitHub 联系我~",
}
