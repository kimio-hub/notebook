export const locales = ["zh", "en"] as const
export type Locale = (typeof locales)[number]

export const defaultLocale: Locale = "zh"

export const localeLabels: Record<Locale, string> = {
  zh: "中文",
  en: "EN",
}

export const htmlLang: Record<Locale, string> = {
  zh: "zh-CN",
  en: "en",
}
