# Bibliography Staging Area

This directory prepares citation metadata before the IEEE MSN author kit and the
ResearchPilot G.0 manuscript directory are initialized. It is not the final
manuscript bibliography.

## Current scope

1. Core mechanism and evaluation references already used by the research
   documents: PaAno, PAI, TSB-AD, Quo Vadis, DADA, TimesURL, SoftCLT, and No
   More Shortcuts.
2. Every distinct model represented in PaAno Tables 2 and 3, including models
   that occur in only one of the U/M tables. A single paper entry is shared by
   aliases such as DLinear/NLinear and MOMENT-FT/MOMENT-ZS.
3. The frozen PaAno code revision, because the paper--code execution comparison
   requires both a publication citation and an immutable software citation.

The candidate file intentionally does not import all 80 entries from PaAno's
source bibliography. ResearchPilot G.5 may add verified references that are
actually needed by the final Related Work section. Before release, every
`\\cite{...}` key must exist, every retained entry must be used, and metadata
must be checked against a primary proceedings, publisher, DOI, OpenReview, or
arXiv source.

## Recency policy

The narrative Related Work is frozen to the 10 keys listed in
`RELATED_WORK_CITATION_PLAN.md`: eight verified 2025--2026 papers and two
indispensable 2024 papers. Thus, 2025--2026 supplies 80% of the narrative set.
Seven of the eight recent papers
are peer reviewed, while PAI is retained with an explicit preprint qualifier.

The two 2024 exceptions are limited to the closest pair-semantics precedents.
The 2024 benchmark/evaluation authorities are cited only in the experimental
protocol, metric, or evidence-policy text and do not enter Related Work.
Any later 2024 narrative addition requires a written indispensability reason
and must preserve at least the 4:1 recent-to-2024 ratio. Papers from 2023 or
earlier cannot be used for narrative novelty
framing. They are retained only for irreplaceable provenance: an original
comparison method, the VUS metric, the inherited RevIN module, or a strict
evaluation rule. Such provenance citations and older table identities do not
count toward the narrative balance.

`dave2024nomore` is staged only as optional cross-domain mechanism provenance;
it is not part of the frozen Related Work narrative and must not be described
as TSAD prior art.

## Result-attribution rule

Model papers establish the identity and design of a comparator. Numerical
values copied from PaAno's benchmark tables remain **PaAno-paper-reported** and
must cite PaAno (and TSB-AD where appropriate); citing a model's original paper
does not turn those values into a local reproduction. The manuscript must not
claim matched hardware, paired testing, or same-code reproduction for those
external rows.

## Source hierarchy

1. Official proceedings or publisher BibTeX.
2. DOI/Crossref metadata.
3. OpenReview or arXiv metadata for papers without proceedings metadata.
4. PaAno arXiv-v3 source bibliography only as an acronym-to-paper mapping aid.

Downloaded PDFs and source archives stay outside Git. Canonical source URLs
are recorded in the source manifest; verified local asset hashes are recorded
in `docs/dev_log.md`.
