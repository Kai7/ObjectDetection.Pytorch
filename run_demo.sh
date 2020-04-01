
#DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577682964_280"
DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577441618_280"

python demo.py --resume saved/thermal_RetinaNet-Tiny_5_final.pt --dataset thermal --demo_path $DEMO_DIR
