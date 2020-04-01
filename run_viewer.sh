PT_PATH=saved/thermal_RetinaNet-Tiny_5_final.pt

python3 view_pytorch_dynamic_info.py --architecture RetinaNet-Tiny \
    --resume $PT_PATH --name retinanet-tiny --in_size 64,80
