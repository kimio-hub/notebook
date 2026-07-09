import { getCollection, type CollectionEntry } from "astro:content"

export type NoteEntry = CollectionEntry<"notes">

export function noteHref(entry: NoteEntry): string {
  // loader ids look like "basics/PPO" or "papers/DeepSeek-V3"
  return `/notes/${entry.id.replace(/\/index$/, "")}/`
}

export function formatDate(date?: Date): string {
  if (!date) return ""
  return date.toLocaleDateString("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

export function formatDateShort(date?: Date): string {
  if (!date) return ""
  return date.toISOString().slice(0, 10)
}

export function categoryLabel(category: string): string {
  if (category === "basics") return "基础"
  if (category === "papers") return "论文"
  return "其他"
}

export async function getPublishedNotes(): Promise<NoteEntry[]> {
  const notes = await getCollection("notes", ({ data, id }) => {
    if (data.draft) return false
    // skip folder markers / index files if any
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
