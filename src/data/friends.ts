export type Friend = {
  name: string
  url: string
  description: string
  avatar?: string
  tags?: string[]
}

/**
 * 友链列表 — 按需编辑本文件后重新构建即可。
 * avatar 可填图片 URL；不填则使用首字母占位。
 */
export const friends: Friend[] = [
  {
    name: "Fried's Blog",
    url: "https://www.friedcali.top/",
    description: "旅行、项目与生活记录，排版干净，更新勤快。",
    tags: ["博客", "项目"],
  },
]

export const friendLinkGuide = {
  title: "申请友链",
  intro: "欢迎互换友链。本站优先与个人独立博客、学习笔记站、研究主页交换。",
  requirements: [
    "有一定原创内容，可长期访问",
    "内容健康，无大量广告或恶意脚本",
    "最好已添加本站链接（可先申请后补）",
  ],
  mine: {
    name: "Kimio",
    url: "https://hides.cc.cd",
    description: "计算机 · RL / LLM 学习笔记与个人主页",
    avatar: "https://hides.cc.cd/avatar.jpg",
  },
  contact: "在 GitHub 提 Issue，或通过站内留下的 GitHub 主页联系我。",
}
