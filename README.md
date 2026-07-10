# Kimio 个人网站

基于 **Astro** 的静态个人站：首页动态、笔记卡片、可视化更新时间线、友链。

- 站点：https://hides.cc.cd
- 仓库：https://github.com/kimio-hub/notebook

## 页面

| 路径 | 说明 |
|------|------|
| `/` | 首页：简介、最近更新、精选笔记 |
| `/notes/` | 全部笔记 |
| `/notes/basics/` | 基础笔记 |
| `/notes/papers/` | 论文精读 |
| `/updates/` | 可视化更新时间线 |
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

- **笔记源目录**：`.env.local` 中的 `SOURCE_NOTES_DIR`（默认指向本地 `论文笔记`）
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
