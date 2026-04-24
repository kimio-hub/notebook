import { access, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises"
import path from "node:path"

const siteRoot = path.resolve(import.meta.dirname, "..")
const contentRoot = path.join(siteRoot, "content")
const outputRoot = path.join(contentRoot, "notes")
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
        if (!trimmed || trimmed.startsWith("#")) {
          continue
        }
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

const wikiLinkPattern = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g

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
    if (entry.isFile()) {
      files.push(fullPath)
    }
  }

  return files
}

async function ensureFrontmatter(content, title) {
  if (content.startsWith("---\n")) {
    return content
  }

  return `---\ntitle: ${title}\n---\n\n${content}`
}

async function hasMarkdownFiles(dir) {
  try {
    const files = await walk(dir)
    return files.some((file) => file.toLowerCase().endsWith(".md"))
  } catch {
    return false
  }
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
  let markdownCount = 0

  for (const file of selectedFiles) {
    const relative = normalize(path.relative(sourceRoot, file))
    const destination = path.join(outputRoot, relative)
    const destinationDir = path.dirname(destination)
    await mkdir(destinationDir, { recursive: true })

    if (file.toLowerCase().endsWith(".md")) {
      const title = path.basename(file, path.extname(file))
      const original = await readFile(file, "utf8")
      const content = await ensureFrontmatter(original, title)
      await writeFile(destination, content, "utf8")
      publishedNames.set(title.toLowerCase(), relative)
      markdownCount += 1
      continue
    }

    await writeFile(destination, await readFile(file))
  }

  const warnings = []
  const selectedMarkdownFiles = selectedFiles.filter((file) => file.toLowerCase().endsWith(".md"))

  for (const file of selectedMarkdownFiles) {
    const relative = normalize(path.relative(sourceRoot, file))
    const content = await readFile(file, "utf8")
    const matches = content.matchAll(wikiLinkPattern)

    for (const match of matches) {
      const target = match[1].trim().toLowerCase()
      if (!publishedNames.has(target)) {
        warnings.push(`${relative} -> [[${match[1].trim()}]]`)
      }
    }
  }

  console.log(
    `Synced ${selectedFiles.length} files (${markdownCount} markdown) to ${normalize(path.relative(siteRoot, outputRoot))}`,
  )
  if (warnings.length > 0) {
    console.warn("Unpublished wiki-link targets detected:")
    for (const warning of warnings) {
      console.warn(`- ${warning}`)
    }
  }
}

async function main() {
  const sourceDir = await loadSourceDir()

  if (sourceDir && (await hasMarkdownFiles(path.resolve(sourceDir)))) {
    await syncFromSource(sourceDir)
    return
  }

  if (await hasMarkdownFiles(outputRoot)) {
    console.log("SOURCE_NOTES_DIR is unavailable. Using committed content/notes for build.")
    return
  }

  console.error("Missing SOURCE_NOTES_DIR and content/notes is empty, so there is nothing to build.")
  process.exit(1)
}

await main()
