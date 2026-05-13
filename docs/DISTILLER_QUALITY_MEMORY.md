# Distiller Quality Memory

This note preserves the current assessment of the LLM Distiller implementation so future debugging conversations can resume from the same state.

## Checked Research

- `research_id`: `research-2f7784cb5b`
- User query: `在LLM中，什么是 Transformer？`
- SQLite database: `.console_runtime/knowledge/session_knowledge.sqlite3`
- Session status: `completed`
- Current round: `6`

## Database Snapshot

Main row:

- `research_sessions.root_query` is preserved correctly as the original user query.
- `research_sessions.status` is `completed`.
- `research_sessions.current_round` is `6`.

Observed table counts:

| Table | Count |
| --- | ---: |
| `session_sources` | 51 |
| `session_facts` | 45 |
| `session_claims` | 41 |
| `session_evidence` | 42 |
| `session_conflicts` | 0 |
| `session_section_packs` | 4 |
| `session_unresolved_gaps` | 14 |
| `session_claim_fact_links` | 49 |
| `session_claim_evidence_links` | 48 |
| `session_pack_fact_links` | 20 |
| `session_pack_claim_links` | 20 |
| `session_pack_evidence_links` | 18 |

Relationship integrity was good in this run:

- No orphan `claim -> fact` links.
- No orphan `claim -> evidence` links.
- No orphan `pack -> fact` links.
- No orphan `pack -> claim` links.
- No orphan `pack -> evidence` links.

## Current Distiller Capability

The current distiller is partially working as an LLM-style distillation component. It no longer only splits sentences heuristically. It can produce useful Chinese facts, claims, and evidence links from retrieved passages.

Examples of useful facts observed:

- `Transformer 是一种神经网络架构。`
- `Transformer 架构以注意力机制为核心。`
- `Transformer 不依赖循环网络作为主要序列建模机制。`
- `Transformer 模型首次提出于论文《Attention is All You Need》。`
- `Transformer 模型由 Vaswani 等人在 2017 年提出。`
- `Transformer 架构是 LLM 的基础。`
- `自注意力是 Transformer 的核心概念。`

Examples of useful claims observed:

- `来源认为 Transformer 架构是现代大语言模型的基础之一。`
- `来源主张自注意力机制使模型能够并行处理输入数据。`
- `来源主张自注意力机制显著提高了训练速度和效果。`
- `来源报告称 Transformer 模型由 Vaswani 等人在 2017 年的论文《Attention is All You Need》中首次提出。`

The implementation is therefore usable for basic fact, claim, and evidence extraction, but not yet reliable enough for high-quality report generation without additional filtering and validation.

## Main Quality Problems

### 1. Low-quality local fallback output still enters SQLite

Round 1 still contained obvious noise such as:

- `This page is displayed while the website verifies you are not a bot.`
- `If you are a data scientist / machine learning student, and this is useful, let me know.`
- `The title ... was a word play in reference to the Beatles song 'Love is all you need'.`
- `Today we will start with the mother of them all...`

Related claims also wrapped noisy text as source-backed claims:

- `来源表明：Warning: This page maybe requiring CAPTCHA...`
- `来源表明：This page is displayed while the website verifies you are not a bot。`

Interpretation: LLM mode may be active in later rounds, but the local fallback path still accepts noisy passages too easily.

### 2. Evidence grounding is not strict enough

Some facts are not directly supported by their stored evidence quote.

Example pattern:

- Evidence: `A decoder is also an enormous neural net.`
- Produced facts included encoder/input-output claims that are not directly supported by that quote.

Interpretation: LLM output validation needs a grounding gate. A fact or claim should be rejected or downgraded if the quote cannot directly support it.

### 3. Some extracted facts are awkward or too generic

Examples:

- `Transformer是一种机器学习。`
- `研究中进行了进一步分析。`
- `研究中进行了基准分析。`

Interpretation: fact quality filters should reject vague meta facts, incomplete definitions, and low-information research-process statements.

### 4. Retrieval noise still pollutes distillation

Observed noisy or low-relevance sources included:

- CAPTCHA / bot verification pages.
- Article title, URL, and Markdown boilerplate.
- Pages not directly answering what Transformer is in LLMs.
- Chatty article prose and unrelated title references.

Interpretation: distiller needs a pre-LLM input filter. It should reject low-relevance or boilerplate passages before prompting the LLM.

### 5. `section_id` remains incomplete

Many distilled rows still have empty `section_id`.

Impact:

- Section evidence packs become less reliable.
- Writer cannot consistently retrieve section-specific facts, claims, and evidence.
- Frontend coverage and gap display can become misleading.

Interpretation: section assignment should be centralized after distillation, and every accepted fact, claim, and evidence item should get a valid section id.

### 6. Gap generation still accepts noisy hints

Open gaps included noisy hints such as:

- `Coverage gap around hinted topic: Seek high_authority_primary_source`
- `Coverage gap around hinted topic: csdn博客 transformer是什么 _什么是transformer 如何理解transformer`
- `Coverage gap around hinted topic: song jimmy`

Interpretation: gap hint generation needs stronger filtering for URL slugs, source titles, boilerplate, and unrelated phrases.

## Implementation Quality Verdict

Current implementation level: about 60-70%.

What works:

- Real retrieved content can be distilled into structured facts.
- Claims and evidence are being created.
- Fact, claim, evidence, and pack link tables are mostly structurally consistent.
- The output now contains several high-value Transformer facts that match the desired direction.

What does not work well enough yet:

- Noise filtering is too weak.
- Local fallback can still produce bad facts.
- LLM output grounding is too permissive.
- `section_id` assignment is incomplete.
- Gap generation still creates noisy pending topics.
- Writer should not fully trust current distilled rows without quality gates.

## Recommended Next Fix Order

1. Add a strict distiller input filter.
   - Reject CAPTCHA, bot verification, title/URL/Markdown boilerplate, obvious author chatter, and off-topic snippets before LLM distillation.

2. Add a post-LLM grounding validator.
   - Every fact and claim must be directly supported by its evidence quote.
   - Reject unsupported or weakly supported outputs.

3. Tighten local fallback.
   - Only allow high-confidence definition, mechanism, timeline, component, and relation facts.
   - Do not wrap arbitrary sentences as `来源表明：...` claims.

4. Centralize section assignment.
   - Ensure every accepted fact, claim, and evidence item has a valid `section_id`.
   - Avoid leaving empty section ids in SQLite.

5. Clean gap hint generation.
   - Filter URL slugs, source titles, generic search instructions, noisy phrases, and unrelated title references.

6. Then improve writer.
   - Writer should consume only quality-filtered section packs.
   - Avoid using all raw distilled rows directly.

## Short Resume Prompt For New Conversations

Continue from `docs/DISTILLER_QUALITY_MEMORY.md`. The current distiller can extract useful Transformer facts, claims, and evidence, but quality is incomplete. Main issues are noisy local fallback rows, weak evidence grounding, incomplete `section_id`, retrieval noise entering distillation, and noisy gap hints. Next fixes should start with strict distiller input filtering, post-LLM grounding validation, and fallback tightening.
