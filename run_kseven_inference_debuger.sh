set -e 

MY_DATA_PATH="/data_host"
SAMPLE_DATA_PATH="$MY_DATA_PATH/trans/client_thermal/sample_data"
PYTHON_TOOL_PATH="$MY_DATA_PATH/trans/client_thermal/python_tool"

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

INFERENCE_SAMPLE_PATH="${PYTHON_TOOL_PATH}/thermal_sample_data_toy"

INPUT_SHAPE="512,640"

python3 kseven_inference_debuger.py --onnx ${OUT_NAME}.onnx \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE \
    --sample_path $INFERENCE_SAMPLE_PATH \
    --pytorch_inference \
    --onnx_inference \
    --compare_head_tensor
