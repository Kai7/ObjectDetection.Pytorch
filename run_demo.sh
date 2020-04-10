
#DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577682964_280"
DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577441618_280"

python demo.py --architecture "RetinaNet-Tiny" \
    --resume saved/thermal_RetinaNet-Tiny_5_final.pt \
    --num_classes 1 \
    --dataset thermal --demo_path $DEMO_DIR
