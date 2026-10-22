# Why I Check Exact Replay When Debugging INT8 Models

[Read the published article on Medium](https://debuggerpaco.medium.com/why-i-check-exact-replay-when-debugging-int8-models-67e67cffc0e6) · Published October 22, 2026.

While testing an INT8 model, I found that the original classifier and its replay gave different output codes for the same image. Both predicted horse, but the horse output code was 16 in the original and 17 in the replay.

Replay here means running the final classifier separately on the same saved integer features, with the same weights, integer bias, scales and zero points. I used its float-output mode and then applied the original output quantization.

I was using the replay to study the final quantization step. Adding that step back needed to reproduce the original codes exactly, so I had to investigate the mismatch before continuing.

INT8 stores values as integers from −128 to 127. The scale sets the size of each step, and the zero point tells us which code represents zero.

In the integer reference, we subtract the zero points, multiply matching feature and weight codes, add the products, and add the integer bias. The scales then convert this sum into a score.

For the horse output, the products summed to 1,427,687. Adding the integer bias of −1,191 gave 1,426,496. Python arithmetic, NumPy arithmetic, and a separate integer operation in the runtime all agreed. The reference sums stayed within INT32 limits.

Using the stored scales without intermediate rounding gave a reference score of about 8.7308830035. The float-output replay gave about 8.7308835983. Float32 rounding can produce slightly different scores.

Quantizing either of these scores using exact division and ties-to-even rounding gives code 17. The small score difference alone does not explain the 16-versus-17 mismatch.

I chose the four candidate scale calculations before running this diagnostic. In the folded-scale paths, I combined the input and weight scales with division by the output scale to make one multiplier. The output scale was about 0.0904754698, and the zero point was −80.

Two candidate paths folded the output scale into the multiplier and gave 96.5 with float32 rounding. Since 96.5 is halfway between 96 and 97, ties-to-even chose 96. Adding the zero point of −80 gave code 16.

Dividing the float-output score by the output scale gave an exact reference slightly above 96.5. Rounding gave 97, and adding the zero point of −80 gave code 17. The standalone quantizer returned 17 for the same score.

![Four rows compare the horse output: S2 lies just below the midpoint, S3 and S4 lie exactly on it, the exact scaled reference lies just above it, and the observed float-output score after exact division lies farther above it. The upper two rows give code 16; the lower two give code 17.](https://raw.githubusercontent.com/gokul-debugger/INT8ExactReplay/v1.0.0/figures/rounding-boundary.png)

*Figure 1. Horse output for test image 4054. Zero on the horizontal axis means 96.5 before integer rounding and zero-point addition; distances are in millionths of a code unit. The upper two rows give code 16; the lower two give code 17. These are reference or candidate calculations, not hidden values observed inside the fused classifier. Both recorded paths still predict horse.*

I used one existing INT8 version of a CIFAR-10 MobileNetV2 x0.5 model. I checked 32 inputs used for earlier controls, plus test image 4054, chosen because its replay had failed. Each input had ten class outputs, giving 33 inputs and 330 output codes.

Repeating each path reproduced its own saved outputs exactly for all 33 inputs; the known 16-versus-17 difference between the paths remained. Turning runtime profiling on or off also left the outputs unchanged. I checked the integer sums with Python integers, NumPy int64 arithmetic, and the runtime's MatMulInteger operation followed by an INT32 bias addition. The runtime combines the original classifier's arithmetic into one operation, called a fused operator.

I ran the checks locally on an Apple M1 Pro Mac with 16 GB RAM, using Python 3.12.13, NumPy 2.5.4 and ONNX Runtime 1.31.0. The model ran on the CPU, processing one input at a time. I used sequential execution, with the intra-operation and inter-operation thread counts both set to one, and enabled all graph optimizations.

Both folded-scale candidates matched all 330 original output codes. A third candidate, which applied the input and weight scales one after the other, also matched all 330 codes.

Different calculation orders can give the same rounded codes. Several paths matched the original output codes, but without the fused classifier’s intermediate values, we couldn’t confirm its calculation order.

The float-output replay runs the classifier in a different output mode; it does not show a hidden score from inside the original quantized calculation. The separate integer check also does not reveal the fused classifier's internal sum. I chose the horse image because its replay had failed. These checks help investigate that failure, but do not measure overall accuracy or show that other models and computers will behave the same way.

Different scale calculations and rounding reproduced the one-code difference using the same checked integer sum. Both paths still predicted horse, but we couldn’t confirm the fused classifier’s calculation order.

The larger study needed the replay to reproduce the original codes after adding final quantization back. Since it didn’t always match, we need to revise the reference and validate it across all models and inputs before continuing.

The [companion code and recorded data](https://github.com/gokul-debugger/INT8ExactReplay/tree/v1.0.0) are available in a separate public repository. Readers can regenerate the arithmetic tables and figure from saved numerical observations, and optionally rerun the standalone output quantizer. The bundle does not include the full model, images, captured features or weight matrices, so it cannot rerun the original classifier or independently reconstruct its integer dot products. The [methods](https://github.com/gokul-debugger/INT8ExactReplay/blob/v1.0.0/METHODS.md) and [limitations](https://github.com/gokul-debugger/INT8ExactReplay/blob/v1.0.0/LIMITATIONS.md) explain that boundary. A [versioned ZIP and checksum](https://github.com/gokul-debugger/INT8ExactReplay/releases/tag/v1.0.0) are included in the release.

When an experiment relies on exact replay, we need to check the output codes. Small differences in arithmetic precision can change them even when the predicted class stays the same.
