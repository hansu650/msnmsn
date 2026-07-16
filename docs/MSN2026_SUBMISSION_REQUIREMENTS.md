# IEEE MSN 2026 Submission Requirements

> Verified from venue- and IEEE-official sources on 2026-07-16. This file is
> the venue-format authority for the manuscript. The generic local paper
> workflow remains an auxiliary writing and build checklist only.

## Venue and Track

- Venue: The 22nd International Conference on Mobility, Sensing and Networking
  (MSN 2026), Ningbo, China, 18--20 December 2026.
- Submission category: Regular Paper.
- Track: **Big Data and AI**.
- Submission system: [MSN 2026 EasyChair](https://easychair.org/conferences/?conf=msn2026).

## Review Manuscript

- IEEE Computer Society Proceedings Format.
- Double column, 10-point font, US Letter paper, submitted as PDF.
- Double-blind review; author-identifying information must be hidden.
- Maximum **8 pages including references and every appendix**.
- The abstract has no formal venue limit, but the CFP states that it is
  usually below 200 words.
- The manuscript must not contain previously published material and must not
  be simultaneously under review elsewhere.

The accepted camera-ready paper may contain up to 10 pages, with an additional
fee for extra pages. The review draft is nevertheless planned against the
strict eight-page inclusive limit.

## Dates

All submission deadlines use Anywhere on Earth (AoE, UTC-12).

| Milestone | Date |
|---|---:|
| Regular-paper submission | 2026-08-20 |
| Acceptance notification | 2026-10-16 |
| Camera-ready submission | 2026-11-07 |
| Registration deadline | 2026-11-17 |
| Conference | 2026-12-18 to 2026-12-20 |

## LaTeX Authority

The manuscript must use IEEE conference format, not Springer LNCS. The source
will therefore use:

```latex
\documentclass[conference]{IEEEtran}
...
\bibliographystyle{IEEEtran}
```

The generic workflow's `llncs.cls`, `splncs04.bst`, `\titlerunning`, and
`\institute` examples are explicitly inapplicable.

The MSN CFP links a 2024 IEEE conference-template ZIP over plain HTTP. The
venue package was independently downloaded, inspected, and compiled with
Tectonic on 2026-07-16. The HTTPS variant returns 404; this explains the
earlier false-unavailable result. The venue kit is the manuscript-template
authority, while the verified CTAN package supplies the bibliography style
and remains a standards-compatible fallback:

| Asset | Local path | SHA-256 |
|---|---|---|
| MSN 2026 venue author kit | `C:/Users/qintian/Downloads/IEEE-MSN-2026-author-kit.zip` | `DCE5B5F34EF738CECE3A86A336795394CB06C2345F79E79B2D456F3D61EC9B9F` |
| IEEEtran CTAN ZIP | `C:/Users/qintian/Downloads/IEEEtran-ctan.zip` | `E0CD4F5AFBD42C8076092280E72B3E0A5111EFE501D35DE9F715CFB8DA313CB4` |

The venue archive is 856,412 bytes, has a safe seven-file layout, and contains
an IEEEtran V1.8b class plus a conference sample using
`\documentclass[conference]{IEEEtran}`. Its sample compiles successfully with
Tectonic 0.16.9. It does not include `IEEEtran.bst`; the CTAN archive contains
`bibtex/IEEEtran.bst`. The two `IEEEtran.cls` files are byte-identical after
line-ending normalization. Because the venue download is HTTP and has no
publisher-posted checksum, recheck the official link and hash immediately
before external submission; a changed official kit takes precedence.

## Local Build Tools

| Asset | Version | SHA-256 |
|---|---|---|
| `D:/qintian_tools/tectonic/0.16.9/tectonic.exe` | 0.16.9 | `A0A9A5EAF1A940D9A615AD78D35225CA59420C7984576C6402FFFB3E9FB05CEB` |
| `C:/Users/qintian/Downloads/CODEX_NEW_PAPER_WORKFLOW.md` | auxiliary workflow | `2C5BA7A400CFA6C9EB0980668C89E6F07793280BCE8BF654BE4EBAA8E9FFE12E` |

Build intermediates must be written below the repository-level
`.latex-build/` tree, not into the paper source directory. Every release build
must check exit status, undefined citations/references, missing files,
overfull content, exact page count, numerical consistency, and rendered-page
layout.

## Official Sources

- [MSN 2026 Call for Regular Papers](https://ieee-msn.org/2026/cf-papers.php)
- [MSN 2026 venue author kit](http://nit.msn2026.top/static/IEEE-conference-template-062824.zip)
- [MSN 2026 home and submission entry](https://ieee-msn.org/2026/)
- [IEEE conference authoring tools and templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
- [IEEE conference template page](https://www.ieee.org/conferences/publishing/templates.html)
- [CTAN IEEEtran package](https://ctan.org/pkg/ieeetran)
