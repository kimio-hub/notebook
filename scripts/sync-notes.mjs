import { access, mkdir, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"
import path from "node:path"

const siteRoot = path.resolve(import.meta.dirname, "..")
const outputRoot = path.join(siteRoot, "src", "content", "notes")
const configPath = path.join(siteRoot, "publish.config.json")

async function pathExists(target) {
  try {
    await access(target)
    return true
  } catch {
    return false
  }
}

async function loadSourceDir() {
  if (process.env.SOURCE_NOTES_DIR) return process.env.SOURCE_NOTES_DIR

  for (const fileName of [".env.local", ".env"]) {
    const envPath = path.join(siteRoot, fileName)
    try {
      const envContent = await readFile(envPath, "utf8")
      for (const line of envContent.split(/\r?\n/)) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith("#")) continue
        const [key, ...valueParts] = trimmed.split("=")
        if (key === "SOURCE_NOTES_DIR") {
          return valueParts.join("=").trim().replace(/^(["'])(.*)\1$/, "$2")
        }
      }
    } catch {
      continue
    }
  }

  return undefined
}

export const normalize = (value) => value.replaceAll("\\", "/")
const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

export const globToRegExp = (pattern) => {
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

const wikiLinkPattern = /(!)?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g

async function readConfig() {
  return JSON.parse(await readFile(configPath, "utf8"))
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  entries.sort((a, b) => a.name.localeCompare(b.name))
  const files = []

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)))
    } else if (entry.isFile()) {
      files.push(fullPath)
    }
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

function formatDate(date) {
  return date.toISOString().slice(0, 10)
}

function frontmatterBoundary(content) {
  if (!content.startsWith("---\n") && !content.startsWith("---\r\n")) return null
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(content)
  if (!match) throw new Error("Unclosed Markdown frontmatter")
  return { frontmatter: match[1], body: content.slice(match[0].length) }
}

export function normalizeMarkdownHeadings(content) {
  const lines = content.replaceAll("\r\n", "\n").split("\n")
  let inFrontmatter = lines[0] === "---"
  let inFence = false
  let fenceMarker = ""

  return lines
    .map((line, index) => {
      if (inFrontmatter) {
        if (index > 0 && line.trim() === "---") inFrontmatter = false
        return line
      }

      const fence = line.match(/^\s*(`{3,}|~{3,})/)
      if (fence) {
        const marker = fence[1][0]
        if (!inFence) {
          inFence = true
          fenceMarker = marker
        } else if (marker === fenceMarker) {
          inFence = false
          fenceMarker = ""
        }
        return line
      }

      return inFence ? line : line.replace(/^# (?=\S)/, "## ")
    })
    .join("\n")
}

function plainText(value) {
  return value
    .replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/[`*_~]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?，。；：！？])/g, "$1")
    .trim()
}

export function descriptionFromMarkdown(content, maxLength = 160) {
  const parsed = frontmatterBoundary(content)
  const body = parsed?.body ?? content
  const prose = body
    .replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/^\s*#{1,6}\s+.*$/gm, "\n")
  const paragraphs = prose.split(/\n\s*\n/).map(plainText).filter((value) => value.length >= 20)
  const text = paragraphs[0] ?? plainText(prose) ?? plainText(body)
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength).trimEnd()}…`
}

function serializeFrontmatterValue(key, value) {
  if (Array.isArray(value)) {
    return value.length ? `${key}:\n${value.map((item) => `  - ${JSON.stringify(item)}`).join("\n")}` : `${key}: []`
  }
  return typeof value === "string" ? `${key}: ${JSON.stringify(value)}` : `${key}: ${value}`
}

export function injectFrontmatter(content, extra) {
  const parsed = frontmatterBoundary(content)
  if (parsed) {
    const lines = []
    for (const [key, value] of Object.entries(extra)) {
      if (value === undefined || value === "") continue
      const re = new RegExp(`^${key}\\s*:`, "m")
      if (!re.test(parsed.frontmatter)) lines.push(serializeFrontmatterValue(key, value))
    }
    if (lines.length === 0) return content
    return `---\n${parsed.frontmatter.trimEnd()}\n${lines.join("\n")}\n---\n${parsed.body}`
  }

  const fields = Object.entries(extra)
    .filter(([, value]) => value !== undefined && value !== "")
    .map(([key, value]) => serializeFrontmatterValue(key, value))
  return `---\n${fields.join("\n")}\n---\n\n${content}`
}

const wikiKey = (value) => normalize(value).replace(/^\.\//, "").replace(/\.md$/i, "").toLowerCase()

export function buildWikiLinkIndex(relativeFiles) {
  const byPath = new Map()
  const byBasename = new Map()

  for (const relative of relativeFiles) {
    const normalized = normalize(relative).toLowerCase()
    const key = wikiKey(normalized)
    byPath.set(key, normalized)
    const base = path.posix.basename(key)
    const matches = byBasename.get(base) ?? []
    matches.push(normalized)
    byBasename.set(base, matches)
  }

  return { byPath, byBasename }
}

function resolveWikiLink(target, currentRelative, index) {
  const targetKey = wikiKey(target.trim())
  const currentDir = path.posix.dirname(normalize(currentRelative))
  const localKey = wikiKey(path.posix.normalize(path.posix.join(currentDir, targetKey)))

  if (index.byPath.has(localKey)) return index.byPath.get(localKey)
  if (index.byPath.has(targetKey)) return index.byPath.get(targetKey)

  const matches = index.byBasename.get(path.posix.basename(targetKey)) ?? []
  if (matches.length > 1) {
    throw new Error(`Ambiguous wiki link "${target}" in ${currentRelative}: ${matches.join(", ")}`)
  }
  return matches[0]
}

export function rewriteWikiLinks(content, index, currentRelative, onWarning = console.warn) {
  return content.replace(wikiLinkPattern, (full, embed, target, alias) => {
    if (embed) {
      onWarning(`Unsupported Obsidian embed kept unchanged in ${currentRelative}: ${full}`)
      return full
    }

    const relative = resolveWikiLink(target, currentRelative, index)
    const label = (alias ?? target).trim()
    if (!relative) {
      onWarning(`Unresolved wiki link in ${currentRelative}: ${full}`)
      return label
    }

    const slug = normalize(relative).replace(/\.md$/i, "").toLowerCase()
    return `[${label}](/notes/${slug}/)`
  })
}

function categoryFromRelative(relative) {
  const top = normalize(relative).split("/")[0].toLowerCase()
  if (top === "basics" || top === "papers") return top
  return "other"
}

async function commitDirectory(tempRoot, destinationRoot) {
  const suffix = `${process.pid}-${Date.now()}`
  const backupRoot = `${destinationRoot}.backup-${suffix}`
  const hadExisting = await pathExists(destinationRoot)

  if (hadExisting) await rename(destinationRoot, backupRoot)
  try {
    await rename(tempRoot, destinationRoot)
  } catch (error) {
    if (hadExisting && (await pathExists(backupRoot))) await rename(backupRoot, destinationRoot)
    throw error
  }
  if (hadExisting) await rm(backupRoot, { recursive: true, force: true })
}

export async function syncFromSource(sourceDir, options = {}) {
  const config = options.config ?? (await readConfig())
  const destinationRoot = options.outputDir ?? outputRoot
  const includeRules = config.include.map(globToRegExp)
  const excludeRules = (config.exclude ?? []).map(globToRegExp)
  const sourceRoot = path.resolve(sourceDir)
  const allFiles = await walk(sourceRoot)

  const selectedFiles = allFiles.filter((file) => {
    const relative = normalize(path.relative(sourceRoot, file))
    return includeRules.some((rule) => rule.test(relative)) && !excludeRules.some((rule) => rule.test(relative))
  })
  const markdownFiles = selectedFiles.filter((file) => file.toLowerCase().endsWith(".md"))
  if (markdownFiles.length === 0) throw new Error("Publish rules selected no Markdown files; existing notes were kept.")

  const suffix = `${process.pid}-${Date.now()}`
  const tempRoot = `${destinationRoot}.tmp-${suffix}`
  await rm(tempRoot, { recursive: true, force: true })
  await mkdir(tempRoot, { recursive: true })

  const relativeFiles = markdownFiles.map((file) => normalize(path.relative(sourceRoot, file)).toLowerCase())
  const wikiIndex = buildWikiLinkIndex(relativeFiles)
  const written = []

  try {
    for (const file of markdownFiles) {
      const relative = normalize(path.relative(sourceRoot, file))
      const relativeLower = relative.toLowerCase()
      const destination = path.join(tempRoot, relativeLower)
      await mkdir(path.dirname(destination), { recursive: true })

      const original = await readFile(file, "utf8")
      const normalized = normalizeMarkdownHeadings(original)
      const fileStat = await stat(file)
      const content = injectFrontmatter(normalized, {
        title: path.basename(file, path.extname(file)),
        description: descriptionFromMarkdown(normalized),
        category: categoryFromRelative(relative),
        date: formatDate(fileStat.mtime),
        updated: formatDate(fileStat.mtime),
      })
      await writeFile(destination, content, "utf8")
      written.push({ destination, relative: relativeLower })
    }

    for (const item of written) {
      const content = await readFile(item.destination, "utf8")
      const rewritten = rewriteWikiLinks(content, wikiIndex, item.relative)
      if (rewritten !== content) await writeFile(item.destination, rewritten, "utf8")
    }

    for (const [folder, page] of Object.entries(config.folderPages ?? {})) {
      const targetDir = folder ? path.join(tempRoot, folder) : tempRoot
      if (!(await pathExists(targetDir))) {
        console.warn(`folderPages: skipping "${folder}"`)
        continue
      }
      const body = page.description ? String(page.description).replace(/<[^>]+>/g, "") : ""
      await writeFile(
        path.join(targetDir, "_folder.md"),
        `---\ntitle: ${JSON.stringify(page.title)}\ndraft: true\ncategory: ${folder === "papers" ? "papers" : folder === "basics" ? "basics" : "other"}\n---\n\n${body}\n`,
        "utf8",
      )
    }

    await commitDirectory(tempRoot, destinationRoot)
  } catch (error) {
    await rm(tempRoot, { recursive: true, force: true })
    throw error
  }

  console.log(`Synced ${markdownFiles.length} markdown notes to ${normalize(path.relative(siteRoot, destinationRoot))}`)
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

  const legacy = path.join(siteRoot, "content", "notes")
  if (await hasMarkdownFiles(legacy)) {
    console.log("Migrating from content/notes ...")
    await syncFromSource(legacy)
    return
  }

  throw new Error("No notes source found. Set SOURCE_NOTES_DIR in .env.local")
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) await main()
