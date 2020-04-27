#PT_PATH=saved/thermal_RetinaNet-Tiny_5_final.pt
PT_PATH=saved/3s-pocket-thermal-face_RetinaNet_P45P6_final_8.pt

#python3 pytorch2onnx.py --architecture RetinaNet-Tiny \
#    --resume $PT_PATH --name retinanet-tiny --in_size 64,80

python3 pytorch2onnx.py --architecture RetinaNet_P45P6 \
    --resume $PT_PATH --name retinanet_p45p6 --num_classes 2 --in_size 288,384
