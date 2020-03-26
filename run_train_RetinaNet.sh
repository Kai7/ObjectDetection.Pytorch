COCO_DATA_ROOT='/data_host/dataset_zoo_tmp/coco/2017'
FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid'

#python train.py --dataset coco --dataset_root ../coco --depth 50
#python train.py --dataset coco --dataset_root "$COCO_DATA_ROOT" --depth 18
#python train.py --batch_size 16 --dataset coco --dataset_root "$COCO_DATA_ROOT" --depth 18
#python train.py --batch_size 8 --dataset coco --dataset_root "$COCO_DATA_ROOT" --depth 50
#
#python train.py --log --learning_rate 0.0001 --weight_decay 0.00005 --batch_size 16 --epochs 50 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18

#python train.py --log --learning_rate 0.0001 --weight_decay 0.0 --batch_size 12 --epochs 10 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18
#python train.py --log --resume saved/FLIR_RetinaNet-Res18_9.pt --learning_rate 0.00001 --weight_decay 0.0 --batch_size 12 --epochs 25 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18
#python train.py --log --learning_rate 0.00001 --weight_decay 0.000001 --batch_size 12 --epochs 50 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18
#python train.py --batch_size 8 --dataset coco --coco_path "$COCO_DATA_ROOT" --depth 50

#python train.py --log --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 15 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18
#python train.py --log --resume saved/FLIR_RetinaNet-Res18_14.pt --learning_rate 0.00001 --weight_decay 0.00001 --batch_size 12 --epochs 30 --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --depth 18


### Termal Dataset Training ###
python train.py --log --learning_rate 0.0001 --weight_decay 0.00001 --batch_size 12 --epochs 5 --dataset thermal --dataset_root "$THERMAL_DATA_ROOT" --depth 18
