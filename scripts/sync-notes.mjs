import { access, mkdir, readdir, readFile, rm, writeFile, stat } from "node:fs/promises"
import path from "node:path"

const siteRoot = path.resolve(import.meta.dirname, "..")
const outputRoot = path.join(siteRoot, "src", "content", "notes")
const configPath = path.join(siteRoot, "publish.config.json")

async function loadSourceDir() {
  if (process.env.SOURCE_NOTES_DIR) {
    return process.env.SOURCE_NOTES_DIR
  }

  for (const fileName of [".env.local", ".env"]) {
    const envPath = path.join(siteRoot, fileName)
    try {
      await access(envPath)
      const envContent = await readFile(envPath, "utf8")
      for (const line of envContent.split(/\r?\n/)) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith("#")) continue
        const [key, ...valueParts] = trimmed.split("=")
        if (key === "SOURCE_NOTES_DIR") {
          return valueParts.join("=").trim()
        }
      }
    } catch {
      continue
    }
  }

  return undefined
}

const normalize = (value) => value.replaceAll("\\", "/")
const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const globToRegExp = (pattern) => {
  const normalized = normalize(pattern)
    .replace(/\*\*\//g, "::DOUBLE_STAR_DIR::")
    .replace(/\*\*/g, "::DOUBLE_STAR::")
    .replace(/\*/g, "::SINGLE_STAR::")
    .replace(/\?/g, "::QUESTION::")

  const escaped = escapeRegex(normalized)
    .replaceAll("::DOUBLE_STAR_DIR::", "(?:.*/)?")
    .replaceAll("::DOUBLE_STAR::", ".*")
    .replaceAll("::SINGLE_STAR::", "[^/]*")
    .replaceAll("::QUESTION::", ".")

  return new RegExp(`^${escaped}$`)
}

const wikiLinkPattern = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g

async function readConfig() {
  return JSON.parse(await readFile(configPath, "utf8"))
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)))
      continue
    }
    if (entry.isFile()) files.push(fullPath)
  }

  return files
}

async function hasMarkdownFiles(dir) {
  try {
    const files = await walk(dir)
    return files.some((file) => file.toLowerCase().endsWith(".md"))
  } catch {
    return false
  }
}

function formatDate(d) {
  return d.toISOString().slice(0, 10)
}

function injectFrontmatter(content, extra) {
  if (content.startsWith("---\n") || content.startsWith("---\r\n")) {
    const end = content.indexOf("\n---", 4)
    if (end === -1) return content
    const fm = content.slice(4, end)
    const body = content.slice(end + 4)
    const lines = []
    for (const [key, value] of Object.entries(extra)) {
      const re = new RegExp(`^${key}\\s*:`, "m")
      if (!re.test(fm)) {
        if (Array.isArray(value)) {
          lines.push(`${key}:`)
          for (const item of value) lines.push(`  - ${item}`)
        } else if (typeof value === "string") {
          lines.push(`${key}: ${JSON.stringify(value)}`)
        } else {
          lines.push(`${key}: ${value}`)
        }
      }
    }
    if (lines.length === 0) return content
    return `---\n${fm.trimEnd()}\n${lines.join("\n")}\n---${body}`
  }

  const title = extra.title ?? "Untitled"
  const tags = extra.tags ?? []
  const tagBlock = tags.length ? `tags:\n${tags.map((t) => `  - ${t}`).join("\n")}\n` : ""
  return `---\ntitle: ${JSON.stringify(title)}\ncategory: ${extra.category}\ndate: ${extra.date}\n${tagBlock}---\n\n${content}`
}

function rewriteWikiLinks(content, publishedNames) {
  return content.replace(wikiLinkPattern, (full, target, alias) => {
    const key = target.trim().toLowerCase()
    const rel = publishedNames.get(key)
    const label = (alias ?? target).trim()
    if (!rel) return label
    // Astro content ids are lowercased in URLs
    const slug = normalize(rel).replace(/\.md$/i, "").toLowerCase()
    return `[${label}](/notes/${slug}/)`
  })
}

function categoryFromRelative(relative) {
  const top = normalize(relative).split("/")[0]
  if (top === "basics" || top === "papers") return top
  return "other"
}

async function syncFromSource(sourceDir) {
  const config = await readConfig()
  const includeRules = config.include.map(globToRegExp)
  const excludeRules = (config.exclude ?? []).map(globToRegExp)
  const sourceRoot = path.resolve(sourceDir)
  const allFiles = await walk(sourceRoot)

  const selectedFiles = allFiles.filter((file) => {
    const relative = normalize(path.relative(sourceRoot, file))
    const included = includeRules.some((rule) => rule.test(relative))
    const excluded = excludeRules.some((rule) => rule.test(relative))
    return included && !excluded
  })

  await rm(outputRoot, { recursive: true, force: true })
  await mkdir(outputRoot, { recursive: true })

  const publishedNames = new Map()
  const written = []
  let markdownCount = 0

  for (const file of selectedFiles) {
    const relative = normalize(path.relative(sourceRoot, file))
    // Lowercase paths so Windows case-folding won't create duplicate content ids
    const relativeLower = relative.toLowerCase()
    const destination = path.join(outputRoot, relativeLower)
    await mkdir(path.dirname(destination), { recursive: true })

    if (file.toLowerCase().endsWith(".md")) {
      const base = path.basename(file, path.extname(file))
      const original = await readFile(file, "utf8")
      const st = await stat(file)
      const category = categoryFromRelative(relative)
      const withFm = injectFrontmatter(original, {
        title: base,
        category,
        date: formatDate(st.mtime),
        updated: formatDate(st.mtime),
      })
      await writeFile(destination, withFm, "utf8")
      publishedNames.set(base.toLowerCase(), relativeLower)
      written.push(destination)
      markdownCount += 1
      continue
    }

    // skip binary assets from notes source for the content collection
  }

  // second pass: rewrite wikilinks using full map
  for (const destination of written) {
    const content = await readFile(destination, "utf8")
    const next = rewriteWikiLinks(content, publishedNames)
    if (next !== content) await writeFile(destination, next, "utf8")
  }

  const folderPages = config.folderPages ?? {}
  for (const [folder, page] of Object.entries(folderPages)) {
    const targetDir = folder ? path.join(outputRoot, folder) : outputRoot
    try {
      await access(targetDir)
    } catch {
      console.warn(`folderPages: skipping "${folder}"`)
      continue
    }
    // folder index pages are handled by Astro routes, not content collection
    // still write a lightweight marker for local browsing if needed
    const body = page.description ? String(page.description).replace(/<[^>]+>/g, "") : ""
    await writeFile(
      path.join(targetDir, "_folder.md"),
      `---\ntitle: ${JSON.stringify(page.title)}\ndraft: true\ncategory: ${folder === "papers" ? "papers" : folder === "basics" ? "basics" : "other"}\n---\n\n${body}\n`,
      "utf8",
    )
  }

  console.log(
    `Synced ${markdownCount} markdown notes to ${normalize(path.relative(siteRoot, outputRoot))}`,
  )
}

async function main() {
  const sourceDir = await loadSourceDir()

  if (sourceDir && (await hasMarkdownFiles(path.resolve(sourceDir)))) {
    await syncFromSource(sourceDir)
    return
  }

  if (await hasMarkdownFiles(outputRoot)) {
    console.log("SOURCE_NOTES_DIR unavailable. Using existing src/content/notes.")
    return
  }

  // fallback: migrate from legacy Quartz content/notes if present
  const legacy = path.join(siteRoot, "content", "notes")
  if (await hasMarkdownFiles(legacy)) {
    console.log("Migrating from content/notes ...")
    await syncFromSource(legacy)
    return
  }

  console.error("No notes source found. Set SOURCE_NOTES_DIR in .env.local")
  process.exit(1)
}

await main()
