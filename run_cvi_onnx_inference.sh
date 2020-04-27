#!/bin/bash

SAMPLE_FILE="inference_sample/3s-pocket-thermal-face/frame_000004_preprocess.npz"

SAMPLE_OUT_FILE="onnx-ThermalFace_retinanet_p45p6.npz"

ONNX_MODEL_PATH="retinanet_p45p6.onnx"

TENSORS_DUMP_PATH="onnx-ThermalFace_retinanet_p45p6_alltensors.npz"


python cvi_onnx_inference.py --input_file $SAMPLE_FILE \
    --output_file $SAMPLE_OUT_FILE \
    --model_path $ONNX_MODEL_PATH \
    --dump_tensors $TENSORS_DUMP_PATH
