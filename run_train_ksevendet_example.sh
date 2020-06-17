set -e 

DATASET_TYPE="kseven"

## Thermal Face Training
#DATASET_ROOT="/data_host/trans/client_thermal/train_valid/v1"
##INPUT_SHAPE="288,384"
#INPUT_SHAPE="384,384"
##BATCH_SIZE=16
#BATCH_SIZE=8


DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
#DATASET_BASE_NAME="sample_manual_20200515_v1"
#DATASET_NAME="3sTFace_Manual_0515"
DATASET_BASE_NAME="sample_manual_CVI"
DATASET_NAME="3sTFace_Manual_CVI"

KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"

INPUT_SHAPE="512,640"
#INPUT_SHAPE="320,384"

#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
#MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_resnet.yaml"
#MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_resnet_bifpn.yaml"
BATCH_SIZE=16
#BATCH_SIZE=24
DATASET_ROOT="${DATASET_ROOT_BASE}/${DATASET_BASE_NAME}"
python train_ksevendet.py --log --dataset ${DATASET_NAME} \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --model_config $MODEL_CONFIG \
    --learning_rate 0.0001 \
    --weight_decay 0.00005 \
    --batch_size $BATCH_SIZE \
    --epochs 8 \
    --input_shape $INPUT_SHAPE \
    --valid_period 1




#MODEL_CONFIG="config/model/ksevendet_shufflenetv2.yaml"
##MODEL_CONFIG="config/model/ksevendet_shufflenetv2_bifpn.yaml"
##MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_shufflenetv2.yaml"
##MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_shufflenetv2_bifpn.yaml"
#BATCH_SIZE=16
#DATASET_ROOT="${DATASET_ROOT_BASE}/${DATASET_BASE_NAME}"
#python train_ksevendet.py --log --dataset ${DATASET_NAME} \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00005 \
#    --batch_size $BATCH_SIZE \
#    --epochs 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 1

#    --resume "saved/final_weight_0601/3sTFace0515_Manual_ksevendet-shufflenetv2_x1_0-bifpn_8.pt" \



#MODEL_CONFIG="config/model/ksevendet_mobilenetv2.yaml"
##MODEL_CONFIG="config/model/ksevendet_mobilenetv2_bifpn.yaml"
##MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_mobilenetv2.yaml"
##MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_mobilenetv2_bifpn.yaml"
#BATCH_SIZE=12
#DATASET_ROOT="${DATASET_ROOT_BASE}/${DATASET_BASE_NAME}"
#python train_ksevendet.py --log --dataset ${DATASET_NAME} \
#    --dataset_root $DATASET_ROOT \
#    --dataset_type $DATASET_TYPE \
#    --model_config $MODEL_CONFIG \
#    --learning_rate 0.0001 \
#    --weight_decay 0.00005 \
#    --batch_size $BATCH_SIZE \
#    --epochs 12 \
#    --input_shape $INPUT_SHAPE \
#    --valid_period 1

