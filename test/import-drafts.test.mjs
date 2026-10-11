import assert from "node:assert/strict"
import test from "node:test"

import { safeAssetName, stripLeadingTitle } from "../scripts/import-drafts.mjs"

test("turns Obsidian attachment names into URL-safe asset names", () => {
  assert.equal(safeAssetName("Pasted image 20260728150250.png"), "pasted-20260728150250.png")
  assert.equal(safeAssetName("p02_img2_xref27.PNG"), "p02_img2_xref27.png")
  assert.equal(safeAssetName("图 1.jpeg"), "1.jpeg")
})

test("drops only a leading H1 title", () => {
  assert.equal(stripLeadingTitle("# Title\n\nBody\n# Later"), "Body\n# Later")
  assert.equal(stripLeadingTitle("\uFEFF\n# Title\nBody"), "Body")
  assert.equal(stripLeadingTitle("## Section\nBody"), "## Section\nBody")
})
