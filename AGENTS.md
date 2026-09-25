---
context-hierarchy: Layer 0
context-hierarchy-role: Global identity
immutable: false
recommended-context-tokens: 900
---

# Data clock visualisation library [`dataclocklib`]

This library allows the user to create data clock graphs, using the matplotlib Python library.

Data clocks visually summarise temporal data in two dimensions, revealing seasonal or cyclical
patterns and trends over time. A data clock is a circular chart that divides a larger unit of time
into rings and subdivides it by a smaller unit of time into wedges, creating a set of temporal
bins.

This project adheres to [YAGNI](context\reference-standard-yagni.md) principles.

## Markdown frontmatter

Mardown files carry `context-hierarchy`, `context-hierarchy-role` and `immutable` keys, plus
`recommended-context-tokens` where a target is given. Beyond those:

- Budgets are a signal, not an enforced limit. A file that outgrows one is worth a look - it may have
  started doing another layer's job.
- Layer 3 carries tags: [keyword, ...]. immutable: true marks the factory configuration, amended
  deliberately, not in passing.

## Reference

Standards and references live in `context/` as reference-*.md files (Layer 3). See the reference
`context\reference-standard-naming.md` document for more information. Load a file when its subject
is in play, which can be determined by the frontmatter.
