#!/usr/bin/env python3
"""
ONNX 转 RKNN 脚本（支持 fp16 / int8 量化）
"""

import argparse
import sys
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX to RKNN converter")
    parser.add_argument("--onnx_path", required=True, help="输入的 ONNX 模型路径")
    parser.add_argument("--rknn_path", required=True, help="输出的 RKNN 模型路径")
    parser.add_argument("--target_platform", required=True,
                        choices=["rk3588", "rk3566", "rk3568", "rv1103", "rv1106"],
                        help="目标芯片平台")
    parser.add_argument("--quantization", required=True,
                        choices=["fp16", "int8"], help="量化类型")
    parser.add_argument("--input_w", type=int, required=True, help="输入宽度")
    parser.add_argument("--input_h", type=int, required=True, help="输入高度")
    parser.add_argument("--input_mean", nargs="+", type=float, default=[0, 0, 0],
                        help="均值，例如 --input_mean 123.675 116.28 103.53")
    parser.add_argument("--input_std", nargs="+", type=float, default=[255, 255, 255],
                        help="标准差，例如 --input_std 58.395 57.12 57.375")
    parser.add_argument("--dataset", help="int8 量化所需的校准数据集文件（每行一张图片路径）")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.quantization == "int8" and not args.dataset:
        print("错误：int8 量化必须提供 --dataset 参数，指向包含图片路径的文本文件。")
        sys.exit(1)

    rknn = RKNN()

    # 配置模型输入
    print(f"目标平台: {args.target_platform}")
    print(f"输入尺寸: {args.input_w} x {args.input_h}")
    print(f"均值: {args.input_mean}")
    print(f"标准差: {args.input_std}")

    ret = rknn.config(
        mean_values=[[args.input_mean]],
        std_values=[[args.input_std]],
        target_platform=args.target_platform
    )
    if ret != 0:
        print(f"配置失败，错误码: {ret}")
        sys.exit(1)

    # 加载 ONNX
    print(f"加载 ONNX: {args.onnx_path}")
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print(f"加载 ONNX 失败，错误码: {ret}")
        sys.exit(1)

    # 构建（量化）
    do_quant = (args.quantization == "int8")
    dataset_arg = args.dataset if do_quant else None
    print(f"开始构建模型，量化类型: {args.quantization}")
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_arg)
    if ret != 0:
        print(f"构建失败，错误码: {ret}")
        sys.exit(1)

    # 导出 RKNN
    print(f"导出 RKNN: {args.rknn_path}")
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print(f"导出失败，错误码: {ret}")
        sys.exit(1)

    print("转换成功！")
    rknn.release()

if __name__ == "__main__":
    main()
