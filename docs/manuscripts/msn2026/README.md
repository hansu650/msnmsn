# Provisional IEEE MSN 2026 Manuscript Scaffold

> Status: `PROVISIONAL_FORMAT_ONLY / PHASE_F_SCIENTIFIC_GATE_PENDING`

This directory is a replaceable formatting scaffold. The CFP-linked MSN 2026
venue kit has since been recovered and recorded in
`docs/MSN2026_SUBMISSION_REQUIREMENTS.md`; its IEEEtran class is equivalent to
the provisional class used here after line-ending normalization. This
scaffold still does not authorize a performance claim or a ResearchPilot Phase
G transition. Scientific prose and result tables remain locked until the
frozen Phase F benchmark gate is resolved.

## Temporary format authority

The recovered venue kit is the format authority. The user-supplied generic
bundle remains a structural reference for the current replaceable scaffold,
pending the user's final template adjustment.

The user supplied the generic IEEE conference bundle at:

`C:/Users/qintian/Downloads/conference-latex-template_10-17-19/Conference-LaTeX-template_10-17-19`

| Asset | SHA-256 | Use |
|---|---|---|
| `conference_101719.tex` | `F701ED9D7BD928FC4A744B3453D1031F89AFED17460F97181819859847A87D8B` | structural reference only |
| `IEEEtran.cls` | `C972ACA108FDA004C3514D63658E02816DA2E54D9A1451E870B9BD970E003F55` | exact, unmodified provisional class |
| `IEEEtran.bst` | `314F0ECE704568FAF827011BAC498650691B2B5EE06320720830E782416D5A5F` | exact CTAN IEEEtran bibliography style |

The class identifies itself as IEEEtran V1.8b (2015-08-26) and uses
`\documentclass[conference]{IEEEtran}`: US Letter, 10-point, two-column
conference layout. The sample prose, sample figure, rendered PDF, HOWTO, and
`.DS_Store` are not copied into this project. The class is retained unmodified
under its LPPL 1.3 notice. The user-supplied 2019 bundle has no `.bst`, so the
bibliography style is copied from the independently verified local
`IEEEtran-ctan.zip` archive.

## Previous-year visual reference

Ten 2025 MSN papers under `C:/Users/qintian/Downloads/bulk-download (4)` were
used only as visual references. All are US Letter and use standard IEEE
two-column typography; nine are eight pages and one is nine pages. Published
conference headers, DOI/copyright strips, page numbers, and IEEE Xplore
download watermarks are production elements and must not be inserted into the
anonymous review source. These papers do not override the 2026 eight-page
review limit recorded in `docs/MSN2026_SUBMISSION_REQUIREMENTS.md`.

## Current source contract

- `main.tex` is anonymous and contains no publisher header, DOI, copyright,
  funding footnote, or author-identifying information.
- `sections/` separates the eventual ResearchPilot G.1--G.6 outputs.
- `references.bib` initially contains only indispensable, already cited
  baseline/benchmark/metric provenance. G.0/G.5 will copy additional used
  entries from `docs/bibliography/candidate_references.bib` and reject unused
  or undefined keys.
- Result placeholders contain no fabricated or incomplete number. They will be
  replaced only from compact artifacts produced after full finalization.
- This dedicated directory contains manuscript sources only. Drafting is
  text-only: edit TeX/BibTeX, prose, equations, and textual tables. Do not
  create figure environments, placeholder figures, image assets, or
  image-generation scripts; the user will add the figures later.
- All build intermediates and provisional PDFs belong only in the
  repository-level `.latex-build/` tree, which is gitignored.

## Local Tectonic build

The verified Tectonic 0.16.9 binary is isolated from both Downloads and the
Git repository at:

`D:/qintian_tools/tectonic/0.16.9/tectonic.exe`

Run builds from the manuscript source directory and direct every generated
file to the repository-level build tree:

```powershell
$paper = 'C:/Users/qintian/Desktop/msn/msnmsn/docs/manuscripts/msn2026'
$out = 'C:/Users/qintian/Desktop/msn/msnmsn/.latex-build/msn2026'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location -LiteralPath $paper
& 'D:/qintian_tools/tectonic/0.16.9/tectonic.exe' `
  --keep-logs --keep-intermediates --outdir $out main.tex
```

The manuscript remains text-only in the current workflow; the source and build
boundaries above remain in force until the user supplies figures.

Before submission, replace or reconcile this scaffold against the user-provided
final MSN template and re-run Tectonic, page-count, citation, numerical, and
rendered-page checks.
