export const site = {
  name: "Kimio",
  title: "Kimio",
  description: "计算机本科生 · 强化学习 / 大语言模型。学习笔记、研究思考。",
  url: "https://hides.cc.cd",
  lang: "zh-CN",
  author: "Kimio",
  email: "dengmail06@163.com",
  github: "https://github.com/kimio-hub",
  githubUser: "kimio-hub",
  zhihu: "https://www.zhihu.com/people/duxingdu-xing",
  role: "计算机 · 强化学习 / 大语言模型",
  tagline: "writing and thinking。",
  bio: "我是 Kimio，计算机专业本科在读。目前主线是强化学习和大语言模型：一边打基础——从 Transformer 到 PPO、GRPO；一边读论文——推理增强、多模态检索、算法博弈论。",
  interests: ["强化学习", "LLM 推理", "多模态检索", "算法博弈论"],
} as const

export type NavItem = {
  label: string
  href: string
}

export const nav: NavItem[] = [
  { label: "首页", href: "/" },
  { label: "笔记", href: "/notes/" },
  { label: "项目", href: "/projects/" },
  { label: "更新", href: "/updates/" },
  { label: "友链", href: "/friends/" },
  { label: "关于", href: "/about/" },
]
