
#DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577682964_280"
#DEMO_DIR="/data_host/trans/client_thermal/dataset/output_1577441618_280"
#DEMO_DIR="/data_host/trans/client_thermal/python_tool/std_annot/train/images"

#DEMO_DIR="/data_host/trans/client_thermal/train_valid/v1/train/images"
DEMO_DIR="/data_host/trans/client_thermal/train_valid/init_sample/train/images"

#python demo.py --architecture "RetinaNet-Tiny" \
#    --resume saved/thermal_RetinaNet-Tiny_5_final.pt \
#    --num_classes 1 \
#    --dataset thermal --demo_path $DEMO_DIR


python demo.py --architecture "RetinaNet_P45P6" \
    --resume saved/3s-pocket-thermal-face_RetinaNet_P45P6_final_8.pt \
    --num_classes 2 \
    --dataset 3s-pocket-thermal-face --demo_path $DEMO_DIR
