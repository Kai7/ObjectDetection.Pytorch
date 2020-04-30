set -e 

#BATCH_SIZE=16

#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="288,384"
#BATCH_SIZE=12
## Thermal Face Training
#python train.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "RetinaNet_P45P6" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 8 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2


## Toy Dataset:Shape Training
DATASET_ROOT="/data_host/dataset_zoo_tmp/shape"
DATASET_TYPE="coco"
INPUT_SHAPE="512,512"
BATCH_SIZE=8
python train.py --log --dataset shape \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --architecture "EfficientDet-D0" \
    --learning_rate 0.001 \
    --weight_decay 0.01 \
    --batch_size $BATCH_SIZE \
    --epochs 50 \
    -j 12 \
    --input_shape $INPUT_SHAPE \
    --valid_period 5

### Toy Dataset:Shape Training
#DATASET_ROOT="/data_host/dataset_zoo_tmp/shape"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="512,512"
#BATCH_SIZE=8
#python train.py --log --dataset shape \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "EfficientDet-D0" \
#    --learning_rate 0.001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 50 \
#    -j 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 1

## Toy Dataset:Shape Training
#python train.py --log --dataset shape \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "RetinaNet_P45P6" \
#    --learning_rate 0.01 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 50 \
#    -j 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 1
