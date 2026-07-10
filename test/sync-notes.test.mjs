import assert from "node:assert/strict"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import test from "node:test"

import {
  buildWikiLinkIndex,
  descriptionFromMarkdown,
  injectFrontmatter,
  normalizeMarkdownHeadings,
  rewriteWikiLinks,
  syncFromSource,
} from "../scripts/sync-notes.mjs"

test("demotes H1 headings outside frontmatter and fenced code", () => {
  const input = `---\ntitle: Demo\n---\n# Section\n\n\`\`\`md\n# code\n\`\`\`\n## Existing`
  const output = normalizeMarkdownHeadings(input)

  assert.match(output, /\n## Section\n/)
  assert.match(output, /```md\n# code\n```/)
  assert.match(output, /\n## Existing$/)
})

test("generates a readable description and preserves an authored one", () => {
  const source = `---\ntitle: Demo\n---\n# Heading\n\nThis is the first useful paragraph with [a link](https://example.com) and **emphasis**.`
  const description = descriptionFromMarkdown(source)
  assert.equal(description, "This is the first useful paragraph with a link and emphasis.")
  assert.equal(
    descriptionFromMarkdown("## 摘要\nA paragraph directly below its heading should be selected.\n\n$$x^2$$"),
    "A paragraph directly below its heading should be selected.",
  )


  const withDescription = injectFrontmatter(source, { description })
  assert.match(withDescription, /^---\ntitle: Demo\ndescription: "This is the first useful paragraph/)

  const authored = injectFrontmatter(`---\r\ntitle: Demo\r\ndescription: Keep me\r\n---\r\nBody`, {
    description: "Replace me",
  })
  assert.match(authored, /description: Keep me/)
  assert.doesNotMatch(authored, /Replace me/)
})

test("resolves local and full-path wiki links while preserving embeds", () => {
  const index = buildWikiLinkIndex(["basics/intro.md", "papers/intro.md"])
  const warnings = []
  const output = rewriteWikiLinks(
    "[[intro]] [[papers/intro|Paper intro]] ![[chart.png]]",
    index,
    "basics/current.md",
    (warning) => warnings.push(warning),
  )

  assert.equal(
    output,
    "[intro](/notes/basics/intro/) [Paper intro](/notes/papers/intro/) ![[chart.png]]",
  )
  assert.equal(warnings.length, 1)
})

test("rejects ambiguous basename-only wiki links", () => {
  const index = buildWikiLinkIndex(["basics/intro.md", "papers/intro.md"])
  assert.throws(
    () => rewriteWikiLinks("[[intro]]", index, "other/current.md", () => {}),
    /Ambiguous wiki link/,
  )
})

test("keeps the previous output when staging fails", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "kimio-sync-failure-"))
  const source = path.join(root, "source")
  const output = path.join(root, "output")
  await mkdir(source, { recursive: true })
  await mkdir(output, { recursive: true })
  await writeFile(path.join(source, "broken.md"), "---\ntitle: Broken\nNo closing frontmatter", "utf8")
  await writeFile(path.join(output, "keep.md"), "keep", "utf8")

  try {
    await assert.rejects(
      syncFromSource(source, {
        outputDir: output,
        config: { include: ["**/*.md"], exclude: [], folderPages: {} },
      }),
      /Unclosed Markdown frontmatter/,
    )
    assert.equal(await readFile(path.join(output, "keep.md"), "utf8"), "keep")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test("commits a fully processed staged output", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "kimio-sync-success-"))
  const source = path.join(root, "source")
  const output = path.join(root, "output")
  await mkdir(path.join(source, "basics"), { recursive: true })
  await mkdir(output, { recursive: true })
  await writeFile(path.join(output, "old.md"), "old", "utf8")
  await writeFile(
    path.join(source, "basics", "demo.md"),
    "# Main section\n\nA useful introductory paragraph that should become the page description.",
    "utf8",
  )

  try {
    await syncFromSource(source, {
      outputDir: output,
      config: { include: ["**/*.md"], exclude: [], folderPages: {} },
    })
    const result = await readFile(path.join(output, "basics", "demo.md"), "utf8")
    assert.match(result, /description: "A useful introductory paragraph/)
    assert.match(result, /\n## Main section\n/)
    await assert.rejects(readFile(path.join(output, "old.md"), "utf8"))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
