import type { Locale } from "../i18n/config"
import { t } from "../i18n/ui"

export const site = {
  name: "Kimio",
  title: "Kimio",
  url: "https://hides.cc.cd",
  author: "Kimio",
  email: "dengmail06@163.com",
  github: "https://github.com/kimio-hub",
  githubUser: "kimio-hub",
  zhihu: "https://www.zhihu.com/people/duxingdu-xing",
} as const

export function siteDescription(lang: Locale) {
  return t(lang, "site.description")
}

export function siteRole(lang: Locale) {
  return t(lang, "site.role")
}

export function siteTagline(lang: Locale) {
  return t(lang, "site.tagline")
}

export function siteBio(lang: Locale) {
  return t(lang, "site.bio")
}

export function siteInterests(lang: Locale) {
  return [
    t(lang, "interest.rl"),
    t(lang, "interest.llm"),
    t(lang, "interest.mm"),
    t(lang, "interest.gt"),
  ]
}

export function homeTitle(lang: Locale) {
  return t(lang, "site.homeTitle")
}
