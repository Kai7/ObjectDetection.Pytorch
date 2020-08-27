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

#MODEL_NAME="3sTFace_resnet18-bifpn-sh_ns-acr_x20_15"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_7.pt"
#OUT_NAME="${MODEL_NAME}-ep_7"
#INPUT_SHAPE="512,640"

#MODEL_NAME="3sTFace_320x384_resnet18-tfdbifpn-sh_ns-acr_x20_10"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-tfdbifpn_final_8.pt"
#OUT_NAME="${MODEL_NAME}-ep_8"
#INPUT_SHAPE="320,384"

#MODEL_NAME="3sTFace_512x640_resnet18-bifpn-sh_ns_3456-acr_x20_20"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_final_8.pt"
#OUT_NAME="${MODEL_NAME}-ep_8"
#INPUT_SHAPE="512,640"

#MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="saved/${DATASET_NAME}_ksevendet-resnet18-bifpn_1.pt"
#OUT_NAME="tmp"
#INPUT_SHAPE="512,640"

MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
TENSOR_PRUNING_DEPENDENCY_JSON_FILE='tensor_pruning_info/ksevendet-resnet18-bifpn-sh_tensor_pruning_dependency.json'
PRUNING_RATE=0.25
RESUME_PT="saved/${DATASET_NAME}_ksevendet-resnet18-bifpn_px75_uniform_6.pt"
ONNX_NAME="thermal_fd_resnet18-bifpn-sh_px75_mAP692"
INPUT_SHAPE="384,512"

python3 pytorch2onnx.py --onnx_name $ONNX_NAME \
    --model_config $MODEL_CONFIG \
    --tensor_dependency $TENSOR_PRUNING_DEPENDENCY_JSON_FILE \
    --pruning_rate $PRUNING_RATE \
    --resume $RESUME_PT \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE 
