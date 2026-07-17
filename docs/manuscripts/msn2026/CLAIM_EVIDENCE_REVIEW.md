# G.7 Claim--Evidence Review

## Decision

The manuscript is internally consistent and ready for an anonymous IEEE MSN submission draft within its stated scope: a frozen, univariate ViT4TS screening-stage audit and repair on 492 series. The evidence supports a narrow coordinate-interface claim. It does not support a general claim about the full VLM4TS system, streaming deployment, or confirmatory post-selection inference.

## Major claims and evidence

| Claim | Direct evidence | Status and boundary |
|---|---|---|
| The released pooled-scale projector is off by one flattened cell. | Exact incidence audit on both pooled masks; every one of the 195 supported queries is displaced, 13 cross a row boundary, and the terminal query is unsupported. | Supported for the released 14 x 14 ViT4TS masks. |
| Literal incidence restores complete grid coverage. | The repaired medium and large incidence matrices each cover 196/196 output cells; their SHA-256 hashes are identical across all 492 cached transactions. | Label-free structural certificate; not inferred from a selected series. |
| IHP changes only the inverse projection. | All 492 arm manifests agree in released cache key, token hash, cache-manifest hash, and encoder-call count. Renderer, encoder, memory, matching, fusion, and stitching are held fixed. | Same-cache causal comparison for the projection operation. |
| IHP improves threshold-free anomaly ranking. | Equal-subdataset AUPRC improves by 0.0218 with 95% interval [0.0062, 0.0398]; VUS-PR improves by 0.0102 with interval [0.0021, 0.0197]. | Supported as unadjusted hierarchical sampling intervals. |
| F1-max improves numerically. | Equal-subdataset F1-max improves by 0.0267, but its interval [-0.0023, 0.0681] crosses zero. | Reported as descriptive, not statistically resolved. |
| IHP exceeds the paper-reported ViT4TS screening row. | IHP obtains 0.662 F1-max versus the external 0.612 value and is numerically higher on 9/11 subdatasets. | External, unpaired context only; the attributable local effect is REL-U to IHP (+0.027). |
| The repair adds no model capacity or encoder pass. | Zero trainable parameters, zero additional encoder calls, zero additional token-cache bytes, and unchanged 11.82 MiB replacement score footprint. | Supported. Arm-isolated runtime, VRAM, and RAM were not measured and are not claimed. |

## High-risk issues resolved in G.7

1. **Novelty overstatement:** the harmonic reducer is explicitly inherited; the new operation is the literal zero-based incidence repair.
2. **Mechanism ambiguity:** the text now distinguishes 195 displaced valid queries, 13 row wraps, and one unsupported terminal query at each pooled scale.
3. **External-comparison ambiguity:** paper-reported ViT4TS and full VLM4TS values are separated from the local same-cache arms and are never used for paired inference.
4. **Selection disclosure:** IHP is identified as a prespecified component promoted after a registered composite arm failed; intervals are explicitly unadjusted for this selection.
5. **Deployment overreach:** full-series preprocessing and all-window median memory are disclosed as offline and transductive, not streaming.
6. **Efficiency overclaim:** only cache, score-storage, encoder-call, and parameter facts are stated; no isolated latency or memory claim remains.

## Five-dimension review

- **Problem and significance:** the paper motivates coordinate-correct localization for telemetry screening and links the failure to a released, executable interface.
- **Technical soundness:** equations define the released shifted incidence, literal repair, inherited harmonic projection, scale fusion, and frozen boundary without hidden trainable choices.
- **Experimental validity:** the primary comparison is paired and same-cache over all 492 series; labels are isolated from score generation; uncertainty follows the benchmark hierarchy.
- **Novelty and positioning:** the contribution is an interface audit plus minimal repair, not a new backbone, harmonic mean, or full multimodal detector.
- **Presentation and reproducibility:** the paper identifies the vendor commit, mirror revision, software/hardware environment, benchmark counts, metrics, seed, structural hashes, and compact artifacts.

## Reverse outline

1. **Introduction:** establishes the inverse-coordinate contract, states the concrete failure, and previews the same-cache evidence.
2. **Related Work:** separates detector/representation advances from inverse-projection correctness and explains the narrower gap.
3. **Method:** formalizes the shifted graph, literal incidence repair, inherited projection, and frozen-system integration.
4. **Experimental Protocol:** defines the 492-series benchmark, local/external comparison boundary, uncertainty, implementation, and label isolation.
5. **Results:** leads with the paired ablation, then external context, structural certificate, and computational boundary.
6. **Limitations and Conclusion:** restates the narrow supported result and discloses selection, offline protocol, and untested generalization.

## Final artifact QA

- Tectonic compilation succeeds.
- Seven US-Letter pages; references begin on page 7.
- No overfull boxes, undefined citations, undefined references, TODOs, or placeholders.
- All 24 cited bibliography entries are defined and used; 19 are from 2025--2026.
- The abstract is below the IEEE 200-word limit.
- The manuscript has 18 embedded fonts and zero raster image XObjects.
- Both supplied figure PDFs are true vector graphics (`image_xobjects = 0`); their Type-3 glyphs are embedded vector outlines.
- All seven rendered pages were visually inspected for clipping, overlap, unreadable tables, and misplaced floats.
- Focused IHP tests pass.
