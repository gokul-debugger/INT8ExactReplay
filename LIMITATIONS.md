# Interpretation limits

- The horse example was chosen after a replay failure. The 32 comparator inputs
  are engineering controls. This is not an unbiased accuracy study or new holdout.
- There is one existing model conversion and one original pretrained checkpoint.
  Agreement here does not establish generality across models, runtimes or hardware.
- Both original and recomposed paths predict horse. A code difference is not
  automatically a changed prediction, reduced accuracy or runtime bug.
- Native float-output mode is a controlled alternative, not a directly observed
  hidden intermediate of quantized QGemm. Its accumulator and pre-round temporary
  remain unobserved. The separate integer probe uses another operator.
- Several arithmetic paths match all 330 original codes. Output agreement alone
  does not uniquely identify native instruction order. Candidate96.5 is an
  emulator output, not a tapped native value.
- Both exact-reference r and observed float-output p map17 under the exact
  quantizer. Their small difference alone is not the 16/17 explanation.
- On the 27 fixed boundary probes, native agrees with exact division25/27,
  float32 division R1 on 27/27 and reciprocal multiplication R2 on 24/27. This
  finite agreement does not prove a universal instruction sequence.
- The exact-rational f32 emulator assigns+0 to exact rational zero, which has
  no sign. The frozen scale paths have positive scales and integer t; observed
  native signed-zero bit patterns remain recorded independently.
- Integer sums and bounds are saved numerical observations in this release.
  Without the excluded feature/weight arrays, their dot products cannot be
  independently reconstructed here. Recorded bound checks are not observations
  of undocumented narrower native intermediates.
- CPU timing/RSS in the original work were budget monitors. This release does
  not establish inference speed, energy or memory savings.
- The larger investigation remains stopped at its original exact gate. This
  bundle covers the arithmetic diagnostic; it does not resume that investigation
  or establish a population accuracy result or an original research method.
