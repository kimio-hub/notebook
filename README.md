# Kimio 个人网站

基于 **Astro** 的静态学术个人主页：侧栏个人信息、动态、精选文章、项目与笔记索引；文章页为衬线阅读栏 + 侧边目录。

- 站点：https://hides.cc.cd
- 仓库：https://github.com/kimio-hub/notebook

## 页面

| 路径 | 说明 |
|------|------|
| `/` | 主页：简介、动态、精选文章、项目、笔记索引 |
| `/notes/` | 全部文章（按年份分组） |
| `/notes/blog/` | 研究长文 |
| `/notes/papers/` | 论文精读 |
| `/notes/courses/` | 课程笔记（CS336） |
| `/notes/basics/` | 基础笔记 |
| `/updates/` | 动态时间线 |
| `/friends/` | 友链 |
| `/about/` | 关于 |

**已移除 RSS。**

## 本地开发

```bash
npm install
npm run dev
```

构建：

```bash
npm run build   # 等同 site:check：同步笔记 + 构建
```

输出目录：`dist/`

## Cloudflare Pages

| 配置项 | 值 |
|--------|-----|
| Build command | `npm run build` |
| Build output directory | `dist` |
| Node version | `22`（或更高） |

## 内容维护

- **笔记源目录**：`.env.local` 中的 `SOURCE_NOTES_DIR`（默认指向本地 `论文笔记`），一级目录决定分类：`blog/`、`papers/`、`courses/`、`basics/`；frontmatter 里的 `category` 优先。
- **发布 learning blog 草稿**：在 `drafts.manifest.json` 里加一条（源文件、目标路径、标题、摘要、标签、分类、图注），运行 `npm run import-drafts`。脚本会生成 frontmatter、去掉开头的一级标题，并把 `![[Pasted image …]]` 和相对路径图片复制到 `public/notes-assets/`。这些文章以 `learning blog` 里的草稿为准：改草稿后重新导入即可（一键更新脚本会自动导入）。
- **精选文章**：frontmatter 加 `feature: true` 即出现在主页「精选文章」。
- **单位 / 所属机构**：`src/i18n/ui.ts` 中的 `site.affiliation`（留空则不显示）。
- **友链**：编辑 `src/data/friends.ts`
- **站点里程碑**：编辑 `src/data/changelog.ts`
- **站点文案**：编辑 `src/data/site.ts`


## 质量检查

```bash
npm run test       # 同步脚本回归测试
npm run typecheck  # Astro 与 TypeScript 检查
npm run site:check # 测试 + 类型检查 + 构建 + 生成物检查
```

`site:check` 会验证所有生成页面只有一个 H1、包含必要元数据，并检查站内链接和图片替代文本。

## 笔记同步保障

同步器先在临时目录完成转换和校验，成功后才替换 `src/content/notes/`；失败时会保留上一次可用内容。同步过程中还会：

- 为缺少 `description` 的笔记提取正文摘要；
- 将正文一级标题降为二级标题，页面标题保持唯一 H1；
- 优先按相对路径解析 Wiki Link，并在同名链接有歧义时终止构建。
一键更新（仓库外脚本）：

```powershell
& "E:\study\workflow\一键更新博客.bat"
```
