PT_PATH=saved/thermal_RetinaNet-Tiny_5_final.pt

python3 pytorch2onnx.py --architecture RetinaNet-Tiny \
    --resume $PT_PATH --name retinanet-tiny --in_size 64,80
