COCO_DATA_ROOT='/data_host/dataset_zoo_tmp/coco/2017'
FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid'
THERMAL_CROSS_DATA_ROOT='/data_host/trans/client_thermal/dataset_cross/thermal_cross'

### Thermal Dataset Training ###
python train.py --log --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" --depth 18

#for i in {0..5};
#do
#    #echo "${THERMAL_CROSS_DATA_ROOT}_${i}"
#    python train.py --log --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 \
#        --dataset thermal --dataset_root "${THERMAL_CROSS_DATA_ROOT}_${i}" --depth 18
#done
