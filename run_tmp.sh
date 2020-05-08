set -e 

# Thermal Face Training
DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
DATASET_TYPE="kseven"
#INPUT_SHAPE="288,384"
INPUT_SHAPE="384,384"
#BATCH_SIZE=16
BATCH_SIZE=8

# Thermal Face Training (Not converge)
MODEL_CONFIG="config/model/ksevendet_efficientnet.yaml"
# for shufflenetv2 & mobilenetv2
python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --model_config $MODEL_CONFIG \
    --learning_rate 0.00001 \
    --weight_decay 0.000001 \
    --batch_size $BATCH_SIZE \
    --epochs 16 \
    --input_shape $INPUT_SHAPE \
    --valid_period 2

    #--optim sgd \
## Thermal Face Training (Not converge)
#MODEL_CONFIG="config/model/ksevendet_mobilenetv2.yaml"
## for shufflenetv2 & mobilenetv2
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.001 \
#    --weight_decay 0.000000 \
#    --batch_size $BATCH_SIZE \
#    --epochs 16 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2

#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
## for shufflenetv2 & mobilenetv2
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 16 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
#
## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_shufflenetv2.yaml"
## for shufflenetv2 & mobilenetv2
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 16 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
#
## Thermal Face Training
#MODEL_CONFIG="config/model/ksevendet_sknet.yaml"
## for shufflenetv2 & mobilenetv2
#python train_ksevendet.py --log --dataset 3s-pocket-thermal-face \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 16 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
#
#


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
#    --model_config ./config/model/ksevendet_shufflenetv2.yaml \
#    --architecture "retinanet-res18" \
#    --learning_rate 0.0001 \
#    --weight_decay 0.000001 \
#    --batch_size $BATCH_SIZE \
#    --epochs 10 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 2
