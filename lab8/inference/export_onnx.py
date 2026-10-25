"""Export the same pretrained ResNet-18 weights as FP32 and FP16 ONNX."""

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
import torch
from torchvision.models import ResNet18_Weights, resnet18


def check_onnx(path, expected_dtype):
    # Passing the filename lets ONNX locate external weight files.
    onnx.checker.check_model(str(path))

    model = onnx.load(str(path), load_external_data=False)
    assert next(op.version for op in model.opset_import if op.domain == "") == 18
    graph = model.graph

    assert len(graph.input) == 1, "Expected one model input"
    assert len(graph.output) == 1, "Expected one model output"

    inp = graph.input[0]
    out = graph.output[0]

    assert inp.name == "input"
    assert out.name == "logits"

    for tensor, expected_tail in (
        (inp, [3, 224, 224]),
        (out, [1000]),
    ):
        tensor_type = tensor.type.tensor_type
        dims = tensor_type.shape.dim

        assert tensor_type.elem_type == expected_dtype, (
            f"Unexpected dtype for {tensor.name}"
        )
        assert len(dims) == len(expected_tail) + 1
        assert dims[0].dim_param, (
            f"Batch dimension is not dynamic: {tensor.name}"
        )
        assert [d.dim_value for d in dims[1:]] == expected_tail

        shape = [d.dim_param or d.dim_value for d in dims]
        dtype_name = onnx.TensorProto.DataType.Name(
            tensor_type.elem_type
        )
        print(f"  {tensor.name}: {shape}, {dtype_name}")

    print(f"  ONNX structure check: PASS ({path})")


def export_one(base_model, precision, dtype, onnx_dtype, output_dir):
    model = deepcopy(base_model).to(dtype=dtype).eval()

    # Only dimension 0 is dynamic; C, H and W stay fixed.
    batch = torch.export.Dim("batch", min=1, max=32)
    example = torch.randn(
        2, 3, 224, 224,
        device="cuda",
        dtype=dtype,
    )

    path = output_dir / f"resnet18_{precision}.onnx"
    print(f"\nExporting {precision}: {path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            dynamic_shapes=({0: batch},),
            external_data=True,
        )

    check_onnx(path, onnx_dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(output_dir / "torch_cache"))
    torch.manual_seed(542)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Pin the weight version for reproducible comparisons.
    weights = ResNet18_Weights.IMAGENET1K_V1
    base_model = resnet18(weights=weights).eval().cuda()

    export_one(
        base_model, "fp32",
        torch.float32, onnx.TensorProto.FLOAT,
        output_dir,
    )
    export_one(
        base_model, "fp16",
        torch.float16, onnx.TensorProto.FLOAT16,
        output_dir,
    )

    print("\nBoth ONNX models exported successfully.")


if __name__ == "__main__":
    main()
