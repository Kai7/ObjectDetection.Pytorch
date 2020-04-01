set -e 

COCO_DATA_ROOT='/data_host/dataset_zoo_tmp/coco/2017'
FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid'
THERMAL_CROSS_DATA_ROOT='/data_host/trans/client_thermal/dataset_cross/thermal_cross'

### Thermal Dataset Training ###
#python train.py --log --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" \
#    --architecture "RetinaNet" --depth 18 \
#    --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 

#python train.py --log --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" \
#    --architecture "RetinaNet-Tiny" --depth 18 \
#    --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 

#for i in {0..5};
#do
#    python train.py --log --dataset thermal --dataset_root "${THERMAL_CROSS_DATA_ROOT}_${i}" \
#        --architecture "RetinaNet-Tiny" --depth 18 \
#        --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 
#done

THERMAL_DATA_ROOT='/data_host/trans/client_thermal/TRAIN'
python TRAIN.py --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" \
    --architecture "RetinaNet-Tiny" --depth 18 \
    --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 
