# 20 — Generating an offline onboarding handbook (PDF)

A new engineer taking over this stack — or you, six months later — needs one thing the scattered concept docs don't give on their own: a **single narrative document** that walks the whole system from first principles to current state, in reading order, as a PDF you can annotate on a couch away from the machine. This doc is the runbook for producing that handbook **entirely offline**, with tools already on the box.

It matters that it is offline. The strictly-local posture ([`01-overview.md`](01-overview.md)) rules out the obvious routes — Google Docs, a hosted Markdown-to-PDF service, any cloud converter — because the handbook, by its nature, quotes internal architecture and (in the private original) infrastructure. The document that explains your private codebase must not be produced by uploading a description of your private codebase. So render it locally.

---

## 1. The pipeline — three steps

```
gather ground truth ──▶ write one self-contained HTML ──▶ render to PDF (headless browser)
  (from the repo,          (print-oriented CSS,             (already installed; no network)
   not from memory)         inline everything)
```

1. **Gather ground truth _from the repo_, not from memory.** A handbook full of stale `path:line` references is worse than none — it teaches the reader wrong. Pull the facts programmatically: module docstrings (`ast.get_docstring`), the DB schemas (`CREATE TABLE …` out of the `.sql` files), the config list, the dependency manifest, the reasoning rules. Cite `file:line`. Everything you assert about the code should have come out of the code minutes earlier.
2. **Write one self-contained HTML file.** Not Markdown-through-a-toolchain you may not have; a single HTML file with an inline `<style>` block. HTML gives you exact control over page breaks, callout boxes, tables, a cover page, and a table of contents — all of which a print stylesheet renders faithfully.
3. **Render to PDF with the headless browser.** A Chromium/Chrome install is almost always already present. Its print engine has excellent CSS support (page breaks, backgrounds, `@page` margins) and runs fully offline. One command turns the HTML into a PDF.

---

## 2. The reusable structure

The handbook's value is as much in its **shape** as its content. This table of contents skeleton has proven to carry a reader from "never seen this" to "can take an issue" — reuse it:

1. **Orientation** — what the system is and is not; the prime directive (strictly-local); the capability ceiling; a one-diagram mental model.
2. **Concept refreshers** — assume a strong engineer who is rusty on the ML specifics. One short section each: language models & the base model; fine-tuning ([`02-fine-tuning.md`](02-fine-tuning.md)); **RAG** as a deep dive ([`08-rag.md`](08-rag.md) — embeddings, vector search, lexical/FTS, fusion, chunking); **the neuro-symbolic layer** as a deep dive ([`14-agent-memory-and-dreaming.md`](14-agent-memory-and-dreaming.md), [`15-deductive-reasoning-and-imagination.md`](15-deductive-reasoning-and-imagination.md) — graph, provenance, the reasoner, imagination); and how tools reach the model ([`07-mcp.md`](07-mcp.md)). These two deep dives are the ones the rest of the system leans on hardest, so spend the words there.
3. **Architecture & code map** — the end-to-end pipeline diagram; a module-by-module table with a "read first" pointer per module; the annotated data-store schemas.
4. **Infrastructure & operations** — the machine roles (the trainer builds; the serving box serves — [`11-hardware.md`](11-hardware.md)); the runbooks ([`17-runbooks.md`](17-runbooks.md)); the local-posture checks.
5. **Evaluation** — the methodology and the results matrix ([`19-evaluating-quality.md`](19-evaluating-quality.md)); this is where a reader learns _which layer to turn on for which job_.
6. **Landmines** — the traps that already bit, each with `file:line` ([`16-lessons-and-fixes.md`](16-lessons-and-fixes.md)). This is the highest-value-per-word section for a new owner.
7. **Current state & roadmap** — what's built, what's designed-not-built, known-open issues.
8. **Appendices** — a glossary ([`12-glossary.md`](12-glossary.md)) and a file/command index.

Scale each section to its reader: dense where it saves them a day (landmines, the reasoning core), light where the code is self-explanatory.

---

## 3. The tooling

### 3.1 The print stylesheet (the reusable core)

The whole look is one inline `<style>`. The load-bearing rules, stripped to essentials:

```css
@page { size: A4; margin: 20mm 18mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; } /* keep background fills in print */
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; }

/* page-break discipline — the single most important part */
.part      { page-break-before: always; }          /* each major part starts a page */
h1,h2,h3   { page-break-after: avoid; }             /* never orphan a heading */
pre, table, .callout { page-break-inside: avoid; }  /* don't split a code block/table across pages */

/* code that WRAPS instead of overflowing the page width */
pre { white-space: pre-wrap; word-break: break-word; overflow: hidden; }

/* a fixed-width diagram box that preserves ASCII art */
.diagram { white-space: pre; font-family: Menlo, monospace; }
```

Add a cover `<section>` sized to the printable height with `page-break-after: always`, a `.toc` section, and simple `.callout` boxes (a coloured left border + tint) for warnings/tips/notes. That's the entire toolkit — no framework.

### 3.2 The render command

```bash
"<chromium-or-chrome-binary>" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="handbook.pdf" "handbook.html"
```

`--no-pdf-header-footer` drops the browser's default URL/date chrome for a clean page. That's it — no network, no service, one binary you already have.

---

## 4. Gotchas

- **Self-contained only.** No web fonts, external stylesheets, CDN scripts, or remote images — both the offline render and the local posture forbid a network fetch. Use **system fonts**; inline any image as a `data:` URI. A single missing external asset silently degrades the whole document.
- **Control page breaks explicitly.** Browsers will happily split a table or a code block across a page boundary. `page-break-inside: avoid` on `pre`/`table`/callouts and `page-break-before` on parts is what makes it read like a book rather than a dump. Long code lines must **wrap** (`white-space: pre-wrap`), or they run off the right edge where print has no horizontal scroll.
- **Verify the output before shipping it.** Don't trust a page-count heuristic — the `file` utility (and similar) often **misreports** the page count of a browser-generated PDF (it compresses the page tree). Open the actual PDF and eyeball the cover, a mid-content page, and the last page. Confirm the diagrams, tables, and callouts survived; confirm nothing got truncated.
- **Ground truth beats prose.** The temptation is to write the architecture section from memory. Don't. Regenerate the docstring/schema/config extracts and paste the real values; a plausible-but-wrong `file:line` is the exact failure mode this handbook exists to prevent. (Same discipline as the rest of this repo: evidence over assertion — [`17-runbooks.md`](17-runbooks.md), "the one rule under all of these".)
- **Keep it private.** An onboarding handbook naturally contains the things you would never publish — machine roles, network layout, credentials-adjacent operational detail. Mark it internal, and deliver it to the reader directly rather than posting it to any shared surface. (The public, teachable version of the same knowledge is what lives in _this_ docs tree — the anonymized split is deliberate.)

---

The handbook is not a substitute for these concept docs; it is the **reading-order narrative** that threads them together for someone starting cold, produced by a method that never leaves the machine. Regenerate it whenever the stack moves enough that the old `file:line` trail has gone stale.
