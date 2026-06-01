#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rknn.api import RKNN


SUPPORTED_PLATFORMS = ["rk3588", "rk3566", "rk3568", "rv1103", "rv1106"]
SUPPORTED_QUANTIZATION = ["fp16", "int8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX to RKNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx_path", required=True, help="Path to the input ONNX model")
    parser.add_argument("--rknn_path", required=True, help="Path to the output RKNN file")
    parser.add_argument(
        "--target_platform",
        required=True,
        choices=SUPPORTED_PLATFORMS,
        help="RKNN target platform",
    )
    parser.add_argument(
        "--quantization",
        required=True,
        choices=SUPPORTED_QUANTIZATION,
        help="Quantization mode",
    )
    parser.add_argument("--input_w", type=int, required=True, help="Model input width")
    parser.add_argument("--input_h", type=int, required=True, help="Model input height")
    parser.add_argument(
        "--input_mean",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("R", "G", "B"),
        help="Input mean for RGB channels",
    )
    parser.add_argument(
        "--input_std",
        nargs=3,
        type=float,
        default=[255.0, 255.0, 255.0],
        metavar=("R", "G", "B"),
        help="Input std for RGB channels",
    )
    parser.add_argument("--dataset", help="Calibration dataset file for int8 quantization")
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    print(f"错误：{message}", file=sys.stderr)
    sys.exit(1)


def validate_file(path: Path, description: str) -> None:
    if not path.is_file():
        fail(f"{description}不存在或不是文件：{path}")


def main() -> int:
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    rknn_path = Path(args.rknn_path)
    dataset_path = Path(args.dataset) if args.dataset else None

    validate_file(onnx_path, "ONNX模型")

    if args.input_w <= 0 or args.input_h <= 0:
        fail("input_w 和 input_h 必须是正整数")

    if args.quantization == "int8":
        if dataset_path is None:
            fail("int8量化必须提供 --dataset")
        validate_file(dataset_path, "int8校准数据集")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"开始转换：{onnx_path} -> {rknn_path}")
    print(f"目标平台：{args.target_platform}，量化模式：{args.quantization}")
    print(f"输入尺寸：{args.input_w}x{args.input_h}")
    print(f"均值：{args.input_mean}，标准差：{args.input_std}")

    rknn = RKNN()
    try:
        ret = rknn.config(
            mean_values=[list(args.input_mean)],
            std_values=[list(args.input_std)],
            target_platform=args.target_platform,
        )
        if ret != 0:
            fail(f"RKNN配置失败，返回码：{ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            fail(f"加载ONNX失败，返回码：{ret}")

        do_quant = args.quantization == "int8"
        ret = rknn.build(
            do_quantization=do_quant,
            dataset=str(dataset_path) if dataset_path else None,
        )
        if ret != 0:
            fail(f"构建RKNN失败，返回码：{ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            fail(f"导出RKNN失败，返回码：{ret}")

        print("转换成功")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    sys.exit(main())
