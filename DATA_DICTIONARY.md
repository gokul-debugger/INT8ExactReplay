# Data dictionary

## data/observations.json

- `parameters`: original feature scale sa,10 per-class weight scales sb, output
  scale sy and integer zero points. Scale records retain decimal display,
  hexadecimal bits/float and exact binary numerator/denominator.
- `samples`:33 records in original order. `ordinal` is0–32; `split`, zero-based
  dataset `index`, label and raw/normalized byte hashes identify the original
  input. No image bytes or normalized image tensors are included.
- `classes`:10 records per sample in the fixed CIFAR-10 class order. `integer`
  contains recorded dot, integer bias, total, partial extrema and absolute bound.
  `native_pre` is the observed float-output score with lossless representation.
  `original_code` and `recomposed_code` are historical native observations.
- `recorded_original_logits`, `recorded_recomposed_logits`: saved native
  dequantized vectors, retained as binary32-roundtrippable JSON numbers.
- `recorded_native_scale_counterfactuals`: S1/S2 native Q/DQ codes/logits from
  the original diagnostic. The arithmetic code treats these as observations.
- `boundary_lower_codes`: the nine prespecified boundary identifiers.

## Other inputs

`boundary-observations.json` contains all 27 boundary values and recorded native
codes/DQ values. The code independently regenerates their exact boundaries,
neighbors, maps and distances. `recorded-controls.json` contains sanitized
historical control summaries, explicitly distinguished from this release's
fresh standalone-quantizer check. No native profiles or absolute paths included.

## Results

- `analysis.json`: full14-path agreement counts and 27-boundary summaries.
- `all-classes.csv`:330 rows with input/class identity, recorded integer sums,
  reference/native scores, signed threshold distances and all 14 code paths.
- `code-match-matrix.csv`:4,620 rows =330×14, with original/recomposed matches.
- `path-decisions.csv`:594 rows =33×18 paths, retaining selected classes/ties.
  These are fixed-cohort arithmetic decisions, not population accuracy estimates.
- `target.json`: all 10 target class records with exact fractions, binary32 bits,
  every S1–S4 intermediate and rounding/clipping flags.
- `boundary-comparison.json`:27 fixed probes and all three reference maps.
- `article-table.md`: the target's code 16/17 reference/candidate comparison.

`*_native` values in saved results are recorded evidence unless the optional
native check is actually run. Regenerated exact references are labeled as such.
All classes and probes remain present, including numerical disagreements.
