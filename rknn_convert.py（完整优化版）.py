#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONNX -> RKNN 转换工具

支持:
    - FP16
    - INT8量化

示例:

FP16:
python rknn_convert.py \
    --onnx_path model.onnx \
    --rknn_path model.rknn \
    --target_platform rk3588 \
    --quantization fp16

INT8:
python rknn_convert.py \
    --onnx_path model.onnx \
    --rknn_path model.rknn \
    --target_platform rk3588 \
    --quantization int8 \
    --dataset dataset.txt
"""

import argparse
import sys
from pathlib import Path

from rknn.api import RKNN


def parse_args():
    """
    解析命令行参数
    """

    parser = argparse.ArgumentParser(
        description="ONNX to RKNN Converter"
    )

    parser.add_argument(
        "--onnx_path",
        required=True,
        help="ONNX模型路径"
    )

    parser.add_argument(
        "--rknn_path",
        required=True,
        help="RKNN输出路径"
    )

    parser.add_argument(
        "--target_platform",
        required=True,
        choices=[
            "rk3588",
            "rk3566",
            "rk3568",
            "rv1103",
            "rv1106"
        ],
        help="目标平台"
    )

    parser.add_argument(
        "--quantization",
        default="fp16",
        choices=["fp16", "int8"],
        help="量化方式"
    )

    parser.add_argument(
        "--input_w",
        type=int,
        help="输入宽度(仅记录)"
    )

    parser.add_argument(
        "--input_h",
        type=int,
        help="输入高度(仅记录)"
    )

    parser.add_argument(
        "--input_mean",
        nargs="+",
        type=float,
        default=[0, 0, 0],
        help="输入均值"
    )

    parser.add_argument(
        "--input_std",
        nargs="+",
        type=float,
        default=[255, 255, 255],
        help="输入标准差"
    )

    parser.add_argument(
        "--dataset",
        help="INT8量化校准集"
    )

    return parser.parse_args()


def validate_args(args):
    """
    参数校验
    """

    if not Path(args.onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX不存在: {args.onnx_path}"
        )

    if len(args.input_mean) != 3:
        raise ValueError(
            "input_mean必须为3个值"
        )

    if len(args.input_std) != 3:
        raise ValueError(
            "input_std必须为3个值"
        )

    if args.quantization == "int8":

        if not args.dataset:
            raise ValueError(
                "INT8模式必须提供dataset"
            )

        if not Path(args.dataset).exists():
            raise FileNotFoundError(
                f"dataset不存在: {args.dataset}"
            )


def print_config(args):
    """
    打印配置
    """

    print("=" * 60)
    print("RKNN Convert Configuration")
    print("=" * 60)

    print(f"ONNX Path       : {args.onnx_path}")
    print(f"RKNN Path       : {args.rknn_path}")
    print(f"Platform        : {args.target_platform}")
    print(f"Quantization    : {args.quantization}")
    print(f"Mean            : {args.input_mean}")
    print(f"Std             : {args.input_std}")

    if args.input_w and args.input_h:
        print(
            f"Input Shape     : "
            f"{args.input_w}x{args.input_h}"
        )

    if args.dataset:
        print(f"Dataset         : {args.dataset}")

    print("=" * 60)


def main():

    args = parse_args()

    try:

        validate_args(args)

        print_config(args)

        rknn = RKNN()

        print("\n[1/4] Config RKNN")

        ret = rknn.config(
            mean_values=[args.input_mean],
            std_values=[args.input_std],
            target_platform=args.target_platform
        )

        if ret != 0:
            raise RuntimeError(
                f"RKNN Config失败: {ret}"
            )

        print("\n[2/4] Load ONNX")

        ret = rknn.load_onnx(
            model=args.onnx_path
        )

        if ret != 0:
            raise RuntimeError(
                f"Load ONNX失败: {ret}"
            )

        do_quant = (
            args.quantization == "int8"
        )

        print(
            f"\n[3/4] Build RKNN "
            f"(quant={args.quantization})"
        )

        ret = rknn.build(
            do_quantization=do_quant,
            dataset=args.dataset if do_quant else None
        )

        if ret != 0:
            raise RuntimeError(
                f"Build失败: {ret}"
            )

        print("\n[4/4] Export RKNN")

        output_dir = Path(args.rknn_path).parent

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        ret = rknn.export_rknn(
            args.rknn_path
        )

        if ret != 0:
            raise RuntimeError(
                f"Export失败: {ret}"
            )

        print("\n✅ 转换成功")
        print(
            f"Output: {args.rknn_path}"
        )

    except Exception as e:

        print(f"\n❌ 错误: {e}")
        sys.exit(1)

    finally:

        try:
            rknn.release()
        except:
            pass


if __name__ == "__main__":
    main()