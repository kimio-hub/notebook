import { getCollection, type CollectionEntry } from "astro:content"
import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"
import { localizePath } from "../i18n/utils"

export type NoteEntry = CollectionEntry<"notes">

export function noteHref(entry: NoteEntry, lang: Locale = "zh"): string {
  const bare = `/notes/${entry.id.replace(/\/index$/, "")}/`
  return localizePath(bare, lang)
}

export function formatDate(date?: Date, lang: Locale = "zh"): string {
  if (!date) return ""
  return date.toLocaleDateString(lang === "en" ? "en-US" : "zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

export function formatDateShort(date?: Date): string {
  if (!date) return ""
  return date.toISOString().slice(0, 10)
}

export function categoryLabel(category: string, lang: Locale = "zh"): string {
  if (category === "basics") return t(lang, "notes.categoryBasics")
  if (category === "papers") return t(lang, "notes.categoryPapers")
  return t(lang, "notes.categoryOther")
}

export async function getPublishedNotes(): Promise<NoteEntry[]> {
  const notes = await getCollection("notes", ({ data, id }) => {
    if (data.draft) return false
    if (id.endsWith("/_folder") || id === "_folder") return false
    if (id.endsWith("/index") || id === "index") return false
    return true
  })

  return notes.sort((a, b) => {
    const da = (a.data.updated ?? a.data.date)?.getTime() ?? 0
    const db = (b.data.updated ?? b.data.date)?.getTime() ?? 0
    return db - da
  })
}

export function excerptFromBody(body?: string, max = 110): string {
  if (!body) return ""
  const text = body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$]+\$/g, " ")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[[^\]]*\]\([^)]+\)/g, (m) => m.replace(/\[|\]|\([^)]*\)/g, ""))
    .replace(/[#>*_`~-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
  if (text.length <= max) return text
  return `${text.slice(0, max).trim()}…`
}
