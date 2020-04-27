SAMPLE_DIR="inference_sample/3s-pocket-thermal-face"

python k7_inference_debuger.py --architecture "RetinaNet_P45P6" \
    --resume saved/3s-pocket-thermal-face_RetinaNet_P45P6_final_8.pt \
    --onnx retinanet_p45p6.onnx \
    --num_classes 2 \
    --dataset 3s-pocket-thermal-face \
    --sample_path $SAMPLE_DIR \
    --pytorch_inference \
    --onnx_inference \
    --compare_head_tensor
    
#    --dump_sample_npz
