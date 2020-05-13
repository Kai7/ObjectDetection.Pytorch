set -e 

DATASET_TYPE="kseven"

## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
##BATCH_SIZE=16
#BATCH_SIZE=8

#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/sample_0"
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/sample_0_1"
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/sample_0_2"

## Thermal Face Training
#DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
#DATASET_ROOT="${DATASET_ROOT_BASE}/sample_0511_clean/sample_0_clean"
#INPUT_SHAPE="512,640"
##BATCH_SIZE=8
#BATCH_SIZE=16
#
## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
#python train_ksevendet.py --log --dataset 3sTFaceSample_0 \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
INPUT_SHAPE="512,640"
BATCH_SIZE=16
MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
for i in {0..4};
do
    DATASET_ROOT="${DATASET_ROOT_BASE}/sample_0511_clean/sample_${i}_clean"
    python train_ksevendet.py --log --dataset 3sTFaceSample_${i} \
        --dataset_root $DATASET_ROOT \
        --dataset_type $DATASET_TYPE \
        --model_config $MODEL_CONFIG \
        --learning_rate 0.0001 \
        --weight_decay 0.00001 \
        --batch_size $BATCH_SIZE \
        --epochs 12 \
        --input_shape $INPUT_SHAPE \
        --valid_period 2
done

## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_shufflenetv2.yaml"
#python train_ksevendet.py --log --dataset 3sTFaceSample_0 \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.00001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 32 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 4

## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_mobilenetv2.yaml"
#python train_ksevendet.py --log --dataset 3sTFaceSample_0 \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.00001 \
#    --weight_decay 0.00001 \
#    --batch_size 12 \
#    --epochs 32 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 4

## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000025 \
#    --batch_size $BATCH_SIZE \
#    --epochs 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_shufflenetv2_bifpn.yaml"
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.00001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 32 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_mobilenetv2_bifpn.yaml"
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.00001 \
#    --weight_decay 0.000001 \
#    --batch_size 12 \
#    --epochs 32 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

#############################################################################
#set -e 
#
##BATCH_SIZE=16
#
## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
#BATCH_SIZE=16
## for mnasnet
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "efficientdet-d0" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 20 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
##BATCH_SIZE=16
#BATCH_SIZE=8
## for shufflenetv2 & mobilenetv2
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "ksevendet" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 20 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="288,384"
##INPUT_SHAPE="384,384"
#BATCH_SIZE=16
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "retinanet-p45p6" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 10 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
#BATCH_SIZE=16
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "retinanet-res18" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 10 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
#DATASET_TYPE="kseven"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
#BATCH_SIZE=16
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "efficientdet-d0" \
#    --learning_rate 0.001 \
#    --weight_decay 0.001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 10 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

### Toy Dataset:Shape Training
#DATASET_ROOT="/data_host/dataset_zoo_tmp/shape"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="512,512"
#BATCH_SIZE=16
#python train_ksevendet.py --log --dataset shape \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "ksevendet" \
#    --learning_rate 0.00001 \
#    --weight_decay 0.00000 \
#    --batch_size $BATCH_SIZE \
#    --epochs 50 \
#    -j 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 5


    #--optim sgd \

### Toy Dataset:Shape Training
#DATASET_ROOT="/data_host/dataset_zoo_tmp/shape"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="512,512"
#BATCH_SIZE=8
#python train_ksevendet.py --log --dataset shape \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "efficientdet-d0" \
#    --learning_rate 0.001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 50 \
#    -j 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 5

### Toy Dataset:Shape Training
#DATASET_ROOT="/data_host/dataset_zoo_tmp/shape"
#DATASET_TYPE="kseven"
#INPUT_SHAPE="512,512"
#BATCH_SIZE=8
#python train_ksevendet.py --log --dataset shape \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --architecture "retinanet-res18" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 50 \
#    -j 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 5

