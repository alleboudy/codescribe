# 21 — Skill selection at scale

This is a **target-architecture** chapter: how a strictly-local coding stack should
discover, choose, and load *skills* when a large codebase — and the many plugins layered
on top of it — contribute thousands of them. It builds on the RAG chapter (§08) and the
data-store layout (handbook §3.3); read those first.

## What a skill is

A **skill** is a small, named procedure a coding agent loads on demand. Concretely it is
a `SKILL.md` file whose YAML frontmatter declares a `name` and a `description` ("use this
when …"), and whose body is a step-by-step recipe for *one* task in your codebase — "add
an API endpoint", "wire a new metric", "scaffold a parser", "cut a release build".

Modern harnesses (claw, and the Claude Code plugin contract it borrows) discover skills
from `.claude/skills/` (and sibling roots) in the working directory **and every
ancestor**, and expose two ways to use one:

- **Operator-invoked** — the human types `/skills <name>`; the harness injects the skill
  body into the prompt as context.
- **Model-invoked** — the model calls a `Skill` tool with a name; the harness loads the
  same body.

Either way the skill's *body* enters context only when it is invoked. That property is
the whole game at scale.

## The property that makes skills scale: lazy loading

A harness does **not** hand the model a menu of every installed skill. Discovery is a
filesystem scan for the operator; the prompt builder injects no skill catalog, and the
model-facing `Skill` tool takes a free-form name with no enumerated list. So whether ten
skills sit on the path or ten thousand, the **standing context cost is identical: zero**.
Only an *invoked* skill's body enters the prompt, for that turn.

```
context cost  =  O(skills invoked this turn)          # a few KB each
              ≠  O(skills installed)                   # this is always zero
```

The practical consequence: **"too many skills" is not a context-window problem.** A team
worried that a big plugin ecosystem will bloat the model's context is worried about the
wrong axis — lazy loading already solved that. The real problem is the mirror image.

## The real problem at scale: selection, not loading

With a huge codebase and many plugins each shipping their own skills, you have hundreds
or thousands of *candidates*. Now:

- The model **can't be shown the full menu** — the descriptions alone would blow the
  context budget you are trying to protect.
- A small, strictly-local model **can't reliably guess** a skill name it was never shown
  (and a weak tool-caller can't be trusted to pick well even if it were).

So the question is never "how do I load a skill" (lazy loading handles that) — it is
**"which skills do I surface for *this* task, cheaply, out of thousands."** That is a
retrieval problem, and it is the one this chapter designs for.

## Three tiers — climb only as far as your corpus forces you

| Corpus size | Mechanism | Standing cost |
|---|---|---|
| ≤ tens | inject the whole menu (name + one-line description) | ~hundreds of tokens/turn |
| ≤ hundreds | **lexical selector** — BM25 over the descriptions | ~0; no embedder, no store |
| thousands, many plugins | **hybrid retrieval + registry + scoping + gate** | one isolated index |

The interface is identical at every tier:

```python
def select(task: str, skills: list[SkillDescriptor], k: int = 5) -> list[ScoredSkill]:
    """Return the top-k skills for a task, most relevant first."""
```

Climbing a tier swaps the *internals* behind `select`; callers — the operator menu, a
`skill_search` tool — never change. **Most teams never pass the lexical tier.** Reach for
vectors only when a single active scope's skill count outgrows lexical recall.

> **Why the lexical tier carries you so far.** Skill descriptions are authored as
> trigger lists ("use this whenever the user wants to add a route, expose a handler,
> scaffold a resource, …"). They are *already* an optimised retrieval surface in prose;
> BM25 over them matches intent well without embeddings. A bespoke skill embedder buys
> little recall here and costs real VRAM — the same constrained memory the served weights
> need (handbook §4, §11). Prefer lexical until measurement says otherwise.

## The target architecture (six layers)

Everything below sits behind `select`. Build the layers you need and no more.

### 1. Registry — aggregate + provenance

Each plugin/source declares its skills; a registry walks all skill roots and builds one
**namespaced** catalog (`plugin:skill`, so two plugins can both ship an `add-endpoint`).
Every descriptor carries, beyond name + description:

- **provenance** — which plugin/root it came from, and its version;
- **trust tier** — first-party · vendored-pinned · third-party;
- **egress class** — does invoking it leave the box (ship a build, post to a chat,
  upload to a store) or stay local?

Provenance and trust are not decoration: a skill body is *executable instruction
context*, so an untrusted plugin's skill is a supply-chain surface (§16, and the same
low-trust posture the harness itself gets — handbook §3.2).

### 2. Index — a **separate, isolated** store

Index the skill *descriptions* in their **own** sqlite-vec + FTS5 collection, distinct
from the code / bug / change indexes (§08, handbook §3.3). Retrieval is **hybrid**: a
BM25/FTS5 lexical prefilter → vector rerank on the survivors.

The isolation is deliberate and load-bearing:

- Skill indexing has a **different lifecycle** (changes when plugins change, not when the
  codebase does) and a **different corpus shape** (many tiny structured descriptors).
- Keeping it in its **own failure domain** means skill work can never corrupt the code
  index or perturb an evaluated retrieval matrix. You *may* reuse the embedder as a
  library; you must **not** share the index.

This is the clean resolution of "no dependency": isolate the **data/index**, optionally
share the **code**. Failure-domain isolation is the guarantee; a duplicated embedder is
just a costly proxy for it.

### 3. Scoping — bound the candidate set

Never search the whole universe. Pre-filter candidates by *active context*:

- enabled plugins,
- working directory + ancestors (the harness already scopes discovery this way),
- trust tier and posture (strictly-local hides network-egress skills by default).

With scoping, even an installation of thousands of skills presents only the few dozen
that are *active and relevant* to any retrieval — so the retriever's job stays bounded no
matter how large the ecosystem grows.

### 4. Selection — a menu, two modes

The top-K descriptors become a **scoped menu**:

- **Operator mode** (primary): `skills select "<task>"` prints a ranked menu; the names
  are what the human hands to the harness as `/skills <name>`. Model-agnostic and
  reliable — it works regardless of how capable the served model is.
- **Model mode**: a `skill_search` MCP tool returns the same top-K so a *capable* model
  can self-select, then load one via `Skill`. Offer it, but do not depend on it when the
  served model is small — selection is exactly where a weak tool-caller is weakest.

Selection surfaces **descriptors only** — never bodies.

### 5. Loading — lazy, on demand

Only the *invoked* skill's body is injected, for that turn. This is the property from the
top of the chapter, and it is the spine every tier keeps: context cost stays proportional
to skills *used*, never skills *installed*.

### 6. Trust + egress gate — posture enforcement

Before a skill is *offered* or *loaded*, check its egress class and trust tier against the
active posture:

- **Strictly-local mode** hides network-egress skills unless the operator explicitly opts
  in — a build-and-upload skill should never be surfaced (much less auto-invoked) in a
  posture whose whole point is that nothing leaves the box.
- **Low-trust plugins** are quarantined or surfaced with a warning, never silently
  promoted into the model's instruction stream.

Refresh is **incremental**: when a plugin updates, re-embed only its changed descriptors,
and follow the external-content FTS5 delete-order rule (§16) so stale trigger terms don't
"ghost" the BM25 scores.

## Why not simply stand up a second vector RAG

A tempting shortcut is "clone the RAG stack for skills." Resist it until scale forces it:

- The **lexical tier already reaches hundreds** of skills with no embedder and no store.
- A second embedder **resident next to the served weights competes for the same VRAM**
  the model needs (handbook §11) — a real cost for a speculative benefit.
- Cloning the full `sources · extract · embed · store · retrieve` pipeline for a corpus of
  tiny descriptors is disproportionate, and *sharing* the code index to avoid the clone
  re-introduces the coupling you were trying to avoid.

When a single active scope genuinely outgrows lexical recall, add a **new collection in
the existing store** (layer 2) — not a parallel subsystem.

## A canonical build order

Stop at whatever tier your corpus needs.

1. **Descriptor parser + registry** — frontmatter → namespaced catalog with provenance +
   egress class. (Target package: `codescribe_train/skills/`.)
2. **Lexical `select(task, k)`** behind the stable interface, plus an operator
   `skills select` CLI that prints a `/skills`-ready menu.
3. **Scoping** — active plugins + cwd/ancestors + posture.
4. **Isolated hybrid index + `skill_search` MCP tool** — *only* when a scope outgrows
   lexical recall.
5. **Trust/egress gate + incremental, ghost-free refresh.**

Steps 1–2 are a few hundred lines of stdlib and cover most real installations. Steps 3–5
are what a genuine multi-plugin, thousands-of-skills ecosystem needs — and the reason to
write the interface down now, so growth is a swap behind `select`, never a rewrite.

## See also

- **§08 — RAG** — the retrieval pipeline whose store and hybrid retriever this reuses.
- **§16 — Lessons & fixes** — the external-content FTS5 delete-order rule (layer 6).
- **Handbook §3.3** — the two data stores; the skill index is a third, isolated one.
- **Handbook §3.5** — the one-screen summary of this chapter, framed as a build target.
