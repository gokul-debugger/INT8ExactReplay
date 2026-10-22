# Methods and reproducibility levels

## Recorded diagnostic

One existing signed-INT8 conversion of a CIFAR-10 MobileNetV2 x0.5 checkpoint was
examined. The target was selected because an earlier exact replay gate failed;
it was not an untouched holdout. The cohort retains32 historical engineering
inputs followed by test image 4054, all 10 output classes:33 inputs,330 codes.

Original CPU conditions: M1 Pro,16 GB RAM, Python 3.12.13, NumPy 2.5.4,
ONNX 1.23.2 and ONNX Runtime 1.31.0; CPUExecutionProvider, batch 1, sequential
execution, intra/inter-op threads 1, ORT_ENABLE_ALL. Timing and RSS were budget
monitors, not performance benchmarks. The original32 controls and target retained
their own saved values bitwise; profiled/unprofiled outputs agreed. The original
model and quantized-head replay returned16 at horse; float-head plus native
requantization returned17. The other9 target codes agreed. Both predict horse.

Two independent Python/int64 integer references and native MatMulInteger+INT32
Add agreed on 330 class sums. Every reference partial sum, bias-added result and
absolute reassociation bound passed INT32 limits. These are historical controls
summarized in `data/recorded-controls.json`, not recomputed from raw features by
this smaller bundle. Native-probe agreement does not expose QGemm's accumulator.

## Arithmetic reproduced here

For class j, the recorded bias-added integer is t. The exact reference is
`r=t*sa*sb[j]`, with exact rational representations of stored binary32 scales.
Every finite binary32 value is encoded with its decimal display, hexadecimal
bits, hexadecimal float and exact numerator/denominator. The original full
precision source bias is not substituted for the recorded integer bias.

`f32` denotes nearest-even binary32 rounding after every marked operation.
No implicit float64 intermediate or contraction is used in the emulator.

| Path | Calculation | Units |
| --- | --- | --- |
| S1 | f32(f32(t)*f32(sa*sb)) | score |
| S2 | f32(f32(f32(t)*sa)*sb) | score |
| S3 | f32(f32(t)*f32(f32(sa*sb)/sy)) | code units before zero point |
| S4 | f32(f32(t)*f32(sa*f32(sb/sy))) | code units before zero point |

For observed p and score-unit S1/S2, compare exact division, float32 division
R1, reciprocal multiplication R2 and recorded native codes. The `*_native`
columns in the arithmetic tables remain recorded observations; use the optional
native script to recheck the standalone Q/DQ maps. Exact quantization is
`clip_INT8(round_even(exact(x)/exact(sy))+zy)`: integer rounding BEFORE zero-point
addition, then clipping. Preserve lowest-index argmax and all ties.

The 14 code paths are p_exact/p_R1/p_R2, r_exact, S1 and S2 each with
exact/R1/R2/native maps, plus S3 and S4. All 330 values per path are retained;
there is no favorable-case filtering.

Nine fixed boundary lower codes are−129,−128,−1,0,15,16,17,126,127. Each exact
boundary is `(q+1/2-zy)*sy`. Use its nearest binary32 value and immediate
predecessor/successor, retaining exact representability and signed distances.
All 27 resulting vectors are retained, including exact/native disagreements.

## Optional native check

The script builds only a two-node standard-ONNX QuantizeLinear/DQ graph with
the original output scale/zero point. It executes33 observed-score vectors,
66 S1/S2 counterfactual vectors and 27 boundary vectors. It requires every
code/DQ value to match the recorded vectors bitwise, checks CPU-only profile
coverage and fails on a changed result. Temporary graphs/profiles are deleted.

## Reproduction boundary

No source images, captured1280-element feature vectors, classifier weight matrix,
full model, upstream model source, calibration memberships or complete experiment
driver is included. The input-row IDs/hashes are provenance identifiers only.
Thus the bundle regenerates arithmetic and rechecks standalone output mapping;
it does not independently recreate original full-model predictions or verify
integer accumulation from feature/weight pairs. Those remain recorded evidence.
Do not describe `python verify.py` as a fresh full-model replication.
