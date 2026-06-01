#!/usr/bin/env python3
import argparse
import sys
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX to RKNN converter")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--rknn_path", required=True)
    parser.add_argument("--target_platform", required=True,
                        choices=["rk3588", "rk3566", "rk3568", "rv1103", "rv1106"])
    parser.add_argument("--quantization", required=True, choices=["fp16", "int8"])
    parser.add_argument("--input_w", type=int, required=True)
    parser.add_argument("--input_h", type=int, required=True)
    parser.add_argument("--input_mean", nargs="+", type=float, default=[0, 0, 0])
    parser.add_argument("--input_std", nargs="+", type=float, default=[255, 255, 255])
    parser.add_argument("--dataset", help="int8校准数据集文件")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.quantization == "int8" and not args.dataset:
        print("错误：int8量化必须提供--dataset")
        sys.exit(1)

    rknn = RKNN()
    ret = rknn.config(
        mean_values=[[args.input_mean]],
        std_values=[[args.input_std]],
        target_platform=args.target_platform
    )
    if ret != 0:
        print(f"配置失败: {ret}")
        sys.exit(1)

    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print(f"加载ONNX失败: {ret}")
        sys.exit(1)

    do_quant = (args.quantization == "int8")
    dataset_arg = args.dataset if do_quant else None
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_arg)
    if ret != 0:
        print(f"构建失败: {ret}")
        sys.exit(1)

    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print(f"导出失败: {ret}")
        sys.exit(1)

    print("转换成功")
    rknn.release()

if __name__ == "__main__":
    main()
