set -e 

KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"

DATASET_NAME="3sTFace_Manual_CVI"

#MODEL_NAME="3sTFace_512x640_resnet18-bifpn-sh_ns_3456-acr_x20_20"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_final_8.pt"

INPUT_SHAPE="512,640"

#RESUME_PT="saved/${DATASET_NAME}_ksevendet-resnet18-bifpn_1.pt"
MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
#MODEL_CONFIG="config/model/ksevendet_shufflenetv2_bifpn.yaml"
#MODEL_CONFIG="config/model/ksevendet_mobilenetv2_bifpn.yaml"
python experiment_model_structure.py \
    --model_config $MODEL_CONFIG \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE 

#    --resume $RESUME_PT \

