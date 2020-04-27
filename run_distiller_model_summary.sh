#!/bin/bash

FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/init_data/train_valid'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/valid'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid/valid'

#MODEL_RESUME_PATH="saved/thermal_RetinaNet-Tiny_4.pth"
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/python_tool/std_annot'

DATA_ROOT='/data_host/trans/client_thermal/python_tool/std_annot'
MODEL_RESUME_PATH="saved/3s-pocket-thermal-face_RetinaNet_P45P6_final_8.pt"

#python distiller_model_summary.py --resume $MODEL_RESUME_PATH --dataset thermal --dataset_root $THERMAL_DATA_ROOT


python distiller_model_summary.py --dataset 3s-pocket-thermal-face \
    --dataset_root $DATA_ROOT \
    --architecture "RetinaNet_P45P6" \
    --num_classes 2 \
    --resume $MODEL_RESUME_PATH 

#python test.py --resume saved/weight/FLIR_RetinaNet-Res18_mAP074_14.pt --dataset FLIR --dataset_root "$FLIR_DATA_ROOT"
