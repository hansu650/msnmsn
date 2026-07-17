# IEEE MSN 2026 IHP Manuscript

> Status: `RESEARCHPILOT_G7_FINAL_REVIEW`

This directory contains the anonymous English IEEE MSN 2026 manuscript for
the final IHP route. The paper is framed as an audit and minimal repair of the
released ViT4TS mask-to-grid coordinate contract. It does not present the
inherited harmonic reducer as a second new module.

## Source map

- `main.tex`: anonymous IEEE conference entry point and paper architecture.
- `IHP_MSN2026_anonymous.pdf`: visually checked seven-page submission draft.
- `sections/`: Abstract, Introduction, Related Work, Method, Experimental
  Protocol, Results, and Limitations/Conclusion.
- `references.bib`: cited bibliography with contribution and citation-reason
  comments for every entry.
- `figures/ihp_method.pdf`: pure-vector mechanism figure.
- `figures/ihp_results.pdf`: pure-vector evidence figure.
- `PAPER_PLAN.md`: ResearchPilot architecture and claim boundaries.
- `CLAIM_EVIDENCE_REVIEW.md`: final G.7 adversarial review and resolution log.

The SVG and 600-DPI PNG figure variants are compatibility fallbacks. The TeX
source uses the vector PDFs, each verified to contain zero image XObjects.

## Evidence boundary

- Primary evidence: paired same-cache `REL_U -> IHP` results on 492 series.
- External evidence: paper-reported ViT4TS values, explicitly descriptive and
  not treated as a local reproduction or paired comparison.
- Selection disclosure: IHP was a prespecified component promoted after a
  composite arm failed on the same evaluation; bootstrap intervals are not
  selection-adjusted.
- Deployment disclosure: released full-series preprocessing and all-window
  median memory make the evaluated screen offline and transductive.
- No claim is made for downstream VLM verification, other backbones,
  multivariate rendering, streaming deployment, or arm-isolated latency/RAM.

Compact numeric evidence lives in `artifacts/ihp/`. Raw data, model weights,
token caches, per-series arrays, failed-route artifacts, and LaTeX build
intermediates are intentionally excluded from the manuscript package.

## Isolated Tectonic build

Run from this directory and keep all generated files under the repository
build tree:

```powershell
$paper = 'C:/Users/qintian/Desktop/msn/msnmsn/docs/manuscripts/msn2026'
$out = 'C:/Users/qintian/Desktop/msn/msnmsn/.latex-build/msn2026-final'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location -LiteralPath $paper
& 'D:/qintian_tools/tectonic/0.16.9/tectonic.exe' `
  --keep-logs --keep-intermediates --outdir $out main.tex
```

Submission QA requires at most eight US-Letter pages, zero undefined
citations/references, zero overfull boxes, embedded fonts, visual inspection
of every rendered page, and numerical agreement with `artifacts/ihp/`.
