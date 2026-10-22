# Upstream sources

Model implementation/checkpoint: chenyaofo/pytorch-cifar-models, source revision
`786c16252c0fc58ee9adac063f8337cc4a7a497a`, MobileNetV2 x0.5 trained for CIFAR-10.
The local pinned source license is BSD 3-Clause, copyright2021 chenyaofo.
The model source, checkpoint and classifier weight matrix are not bundled.

Source/license:
https://github.com/chenyaofo/pytorch-cifar-models/tree/786c16252c0fc58ee9adac063f8337cc4a7a497a
https://github.com/chenyaofo/pytorch-cifar-models/blob/786c16252c0fc58ee9adac063f8337cc4a7a497a/LICENSE

Dataset mirror: uoft-cs/cifar10, revision
`0b2714987fa478483af9968de7c934580d0bb9a2`.
https://huggingface.co/datasets/uoft-cs/cifar10/tree/0b2714987fa478483af9968de7c934580d0bb9a2
The listed indices are zero-based rows in that pinned mirror, not a promise that
all dataset libraries use identical ordering. Images and tensor inputs are not
redistributed, and this bundle does not assign them a license.

Conceptual quantization background: Jacob et al., *Quantization and Training of
Neural Networks for Efficient Integer-Arithmetic-Only Inference*, preprintv1:
https://arxiv.org/abs/1712.05877v1 . No paper text or figures are reproduced.

Dependencies: NumPy, ONNX, ONNX Runtime, and optional Matplotlib. Their source/
binaries are not included. Standard operator semantics are documented by ONNX:
https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
The recorded model uses standard opset17 (QuantizeLinear schema version13),
so do not substitute newer optional operator behavior when interpreting results.

No upstream author or organization endorsement is asserted.
