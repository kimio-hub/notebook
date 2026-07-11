import { defaultLocale, locales, type Locale } from "./config"
import { t } from "./ui"

export function isLocale(value: string | undefined): value is Locale {
  return !!value && (locales as readonly string[]).includes(value)
}

/** Detect locale from pathname. Default zh has no prefix. */
export function getLocaleFromUrl(url: URL): Locale {
  const seg = url.pathname.split("/").filter(Boolean)[0]
  if (isLocale(seg) && seg !== defaultLocale) return seg
  return defaultLocale
}

/** Strip non-default locale prefix (e.g. /en/notes/ → /notes/). */
export function stripLocalePrefix(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean)
  if (parts[0] && isLocale(parts[0]) && parts[0] !== defaultLocale) {
    const rest = parts.slice(1)
    if (rest.length === 0) return "/"
    return `/${rest.join("/")}/`
  }
  if (!pathname || pathname === "/") return "/"
  return pathname.endsWith("/") ? pathname : `${pathname}/`
}

export function ensureTrailingSlash(path: string): string {
  if (!path || path === "/") return "/"
  if (path.includes("://") || path.startsWith("mailto:")) return path
  const [p, q] = path.split("?")
  const withSlash = p.endsWith("/") ? p : `${p}/`
  return q ? `${withSlash}?${q}` : withSlash
}

/** Build a localized path for an unprefixed route like /notes/ or /projects/foo/ */
export function localizePath(path: string, lang: Locale): string {
  if (path.startsWith("http") || path.startsWith("mailto:")) return path
  const bare = stripLocalePrefix(ensureTrailingSlash(path))
  if (lang === defaultLocale) return bare
  if (bare === "/") return `/${lang}/`
  return `/${lang}${bare}`
}

/** Switch current pathname to another locale, preserving the logical route. */
export function switchLocalePath(pathname: string, target: Locale): string {
  return localizePath(stripLocalePrefix(pathname), target)
}

export function getNav(lang: Locale) {
  return [
    { label: t(lang, "nav.home"), href: localizePath("/", lang) },
    { label: t(lang, "nav.notes"), href: localizePath("/notes/", lang) },
    { label: t(lang, "nav.projects"), href: localizePath("/projects/", lang) },
    { label: t(lang, "nav.updates"), href: localizePath("/updates/", lang) },
    { label: t(lang, "nav.friends"), href: localizePath("/friends/", lang) },
    { label: t(lang, "nav.about"), href: localizePath("/about/", lang) },
  ]
}

export { t }
