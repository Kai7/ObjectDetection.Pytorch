set -e 

KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"
WEIGHTS_SAVE_PATH="/data_host/trans/pytorch/ObjectDetection.Pytorch/saved"

DATASET_NAME="3sTFace_Manual_CVI"

#MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0605/${DATASET_NAME}_ksevendet-resnet18-bifpn_final_8.pt"
#OUT_NAME="${DATASET_NAME}_resnet18-bifpn_final_8"

#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
#RESUME_PT="${WEIGHTS_SAVE_PATH}/final_weights_0604/${DATASET_NAME}_ksevendet-resnet18-fpn_7.pt"
#OUT_NAME="${DATASET_NAME}_resnet18-fpn_7"

MODEL_NAME="3sTFace_resnet18-bifpn-sh_ns-acr_x20_15"
MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_7.pt"
OUT_NAME="${MODEL_NAME}-ep_7"

INPUT_SHAPE="512,640"

python3 kseven_pytorch2onnx.py --onnx_name $OUT_NAME \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE 
