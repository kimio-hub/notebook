import { getCollection, type CollectionEntry } from "astro:content"
import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"
import { localizePath } from "../i18n/utils"

export type NoteEntry = CollectionEntry<"notes">
export type NoteCategory = Exclude<NoteEntry["data"]["category"], "other">

/** Display order for categories across the site. */
export const noteCategories: NoteCategory[] = ["blog", "papers", "courses", "basics"]

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
  if ((noteCategories as string[]).includes(category)) return t(lang, `notes.category.${category}`)
  return t(lang, "notes.category.other")
}

/** Rough reading time: ~400 CJK characters or ~220 Latin words per minute. */
export function readingMinutes(body?: string): number {
  if (!body) return 1
  const text = body.replace(/```[\s\S]*?```/g, " ").replace(/\$\$[\s\S]*?\$\$/g, " ")
  const cjk = (text.match(/[㐀-鿿]/g) ?? []).length
  const words = (text.replace(/[㐀-鿿]/g, " ").match(/[A-Za-z0-9]+/g) ?? []).length
  return Math.max(1, Math.round(cjk / 400 + words / 220))
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
    return db - da || a.id.localeCompare(b.id)
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
