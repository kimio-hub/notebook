// Import finished drafts from the "learning blog" folder into the notes source folder.
//
// Each entry in drafts.manifest.json maps one draft to a published note: frontmatter is generated,
// the leading H1 is dropped (the page renders the title), and images — Obsidian `![[...]]` embeds or
// relative Markdown images — are copied into public/notes-assets/<assets>/ with stable names.
// Run `npm run import-drafts`, then `npm run build` (which syncs the notes folder into the site).
import { copyFile, mkdir, readFile, stat, writeFile, access } from "node:fs/promises"
import { fileURLToPath } from "node:url"
import path from "node:path"

const siteRoot = path.resolve(import.meta.dirname, "..")
const manifestPath = path.join(siteRoot, "drafts.manifest.json")
const assetsRoot = path.join(siteRoot, "public", "notes-assets")

async function exists(target) {
  try {
    await access(target)
    return true
  } catch {
    return false
  }
}

function localDate(date) {
  const pad = (n) => String(n).padStart(2, "0")
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
}

export function safeAssetName(fileName) {
  const ext = path.extname(fileName).toLowerCase()
  const stem = path
    .basename(fileName, path.extname(fileName))
    .replace(/^Pasted image /i, "pasted-")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return `${stem || "image"}${ext}`
}

export function stripLeadingTitle(markdown) {
  return markdown.replace(/^﻿/, "").replace(/^\s*# [^\n]*\n+/, "")
}

function frontmatter(entry, date) {
  const lines = [
    `title: ${JSON.stringify(entry.title)}`,
    `description: ${JSON.stringify(entry.description)}`,
    entry.tags?.length ? `tags:\n${entry.tags.map((tag) => `  - ${JSON.stringify(tag)}`).join("\n")}` : "tags: []",
    `category: ${JSON.stringify(entry.category)}`,
  ]
  if (entry.series) lines.push(`series: ${JSON.stringify(entry.series)}`)
  if (entry.feature) lines.push("feature: true")
  lines.push(`date: ${JSON.stringify(date)}`, `updated: ${JSON.stringify(date)}`)
  return `---\n${lines.join("\n")}\n---\n\n`
}

async function findImage(name, searchDirs) {
  for (const dir of searchDirs) {
    const candidate = path.join(dir, name)
    if (await exists(candidate)) return candidate
  }
  throw new Error(`Image not found: ${name} (searched ${searchDirs.join(", ")})`)
}

async function importEntry(entry, manifest) {
  const sourcePath = path.join(manifest.draftsDir, entry.source)
  const sourceDir = path.dirname(sourcePath)
  const assetDir = path.join(assetsRoot, entry.assets ?? path.basename(entry.target, ".md"))
  const assetUrl = `/notes-assets/${entry.assets ?? path.basename(entry.target, ".md")}`
  const captions = entry.captions ?? {}
  const date = entry.date ?? localDate((await stat(sourcePath)).mtime)
  const copies = []

  let body = stripLeadingTitle((await readFile(sourcePath, "utf8")).replaceAll("\r\n", "\n"))

  const embeds = [...body.matchAll(/!\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]/g)]
  for (const match of embeds) {
    const name = match[1].trim()
    const file = await findImage(name, [sourceDir, ...(manifest.attachmentDirs ?? [])])
    const target = safeAssetName(name)
    copies.push([file, target])
    const caption = captions[name]
    const image = `![${caption ?? entry.title}](${assetUrl}/${target})`
    body = body.replace(match[0], caption ? `${image}\n\n*${caption}*` : image)
  }

  body = body.replace(/!\[([^\]]*)\]\((?!https?:|\/)([^)\s]+)\)/g, (_match, alt, relative) => {
    const file = path.join(sourceDir, decodeURI(relative))
    const target = safeAssetName(path.basename(relative))
    copies.push([file, target])
    return `![${alt || entry.title}](${assetUrl}/${target})`
  })

  if (copies.length) await mkdir(assetDir, { recursive: true })
  for (const [file, target] of copies) await copyFile(file, path.join(assetDir, target))

  const destination = path.join(manifest.notesDir, entry.target)
  await mkdir(path.dirname(destination), { recursive: true })
  await writeFile(destination, frontmatter(entry, date) + body.trimEnd() + "\n", "utf8")
  return { images: copies.length }
}

async function main() {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"))
  const only = process.argv.slice(2)
  const entries = only.length
    ? manifest.entries.filter((entry) => only.some((key) => entry.target.includes(key)))
    : manifest.entries

  for (const entry of entries) {
    const { images } = await importEntry(entry, manifest)
    console.log(`Imported ${entry.target}${images ? ` (+${images} images)` : ""}`)
  }
  console.log(`Imported ${entries.length} draft(s) into ${manifest.notesDir}`)
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) await main()
