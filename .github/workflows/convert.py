from rknn.api import RKNN
import os

ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'

rknn = RKNN(verbose=True)

print('=> Config')

rknn.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]]
)

print('=> Load ONNX')

ret = rknn.load_onnx(
    model=ONNX_MODEL
)

if ret != 0:
    raise Exception('Load ONNX failed')

print('=> Build RKNN')

ret = rknn.build(
    do_quantization=False
)

if ret != 0:
    raise Exception('Build RKNN failed')

print('=> Export RKNN')

ret = rknn.export_rknn(RKNN_MODEL)

if ret != 0:
    raise Exception('Export RKNN failed')

print('RKNN saved:', RKNN_MODEL)

rknn.release()
