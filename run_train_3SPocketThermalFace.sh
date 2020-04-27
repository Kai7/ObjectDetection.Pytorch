set -e 

COCO_DATA_ROOT='/data_host/dataset_zoo_tmp/coco/2017'
FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid'
THERMAL_CROSS_DATA_ROOT='/data_host/trans/client_thermal/dataset_cross/thermal_cross'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/python_tool/std_annot'

TRAIN_VALID_DATA_ROOT="/data_host/trans/client_thermal/train_valid/v1"


#for i in {0..5};
#do
#    python train.py --log --dataset thermal --dataset_root "${THERMAL_CROSS_DATA_ROOT}_${i}" \
#        --architecture "RetinaNet-Tiny" --depth 18 \
#        --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 
#done

#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/TRAIN'
#python TRAIN.py --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" \
#    --architecture "RetinaNet-Tiny" --depth 18 \
#    --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 

# Thermal Face Training
python train.py --log --dataset 3s-pocket-thermal-face --dataset_root "$TRAIN_VALID_DATA_ROOT" \
    --architecture "RetinaNet_P45P6" \
    --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 8 --valid_period 2
