import { access, readdir, readFile } from "node:fs/promises"
import path from "node:path"

const distRoot = path.resolve(import.meta.dirname, "..", "dist")

async function exists(target) {
  try {
    await access(target)
    return true
  } catch {
    return false
  }
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) files.push(...(await walk(fullPath)))
    else if (entry.isFile()) files.push(fullPath)
  }
  return files
}

function internalTarget(href) {
  const pathname = href.split("#", 1)[0].split("?", 1)[0]
  if (!pathname.startsWith("/") || pathname.startsWith("//")) return null
  const relative = decodeURIComponent(pathname).replace(/^\/+/, "")
  if (!relative) return path.join(distRoot, "index.html")
  const target = path.join(distRoot, relative)
  return path.extname(target) ? target : path.join(target, "index.html")
}

const files = await walk(distRoot)
const htmlFiles = files.filter((file) => file.endsWith(".html"))
const errors = []

for (const file of htmlFiles) {
  const relative = path.relative(distRoot, file).replaceAll("\\", "/")
  const html = await readFile(file, "utf8")
  const titleCount = (html.match(/<title>[^<]+<\/title>/gi) ?? []).length
  const descriptionCount = (html.match(/<meta\s+name="description"\s+content="[^"]+"/gi) ?? []).length
  const h1Count = (html.match(/<h1(?:\s|>)/gi) ?? []).length
  const imagesWithoutAlt = html.match(/<img(?![^>]*\balt=)[^>]*>/gi) ?? []

  if (titleCount !== 1) errors.push(`${relative}: expected one non-empty <title>, found ${titleCount}`)
  if (descriptionCount !== 1) errors.push(`${relative}: expected one non-empty meta description`)
  if (h1Count !== 1) errors.push(`${relative}: expected one <h1>, found ${h1Count}`)
  if (imagesWithoutAlt.length) errors.push(`${relative}: ${imagesWithoutAlt.length} image(s) missing alt`)

  for (const match of html.matchAll(/href="([^"]+)"/gi)) {
    const target = internalTarget(match[1])
    if (target && !(await exists(target))) errors.push(`${relative}: broken internal link ${match[1]}`)
  }
}

if (errors.length) {
  console.error(`Built-site check failed with ${errors.length} issue(s):`)
  for (const error of errors) console.error(`- ${error}`)
  process.exitCode = 1
} else {
  console.log(`Built-site check passed: ${htmlFiles.length} HTML pages, links and metadata verified.`)
}
