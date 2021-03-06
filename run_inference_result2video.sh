set -e 

#DEMO_DIR="/data_host/trans/client_thermal/train_valid/init_sample/train/images"
#DEMO_DIR="/data_host/trans/client_thermal/train_valid/sample_0/valid/images"
DEMO_DIR="/data_host/trans/client_thermal/train_valid/sample_manual_CVI/valid/images"
KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"
WEIGHTS_SAVE_PATH="/data_host/trans/pytorch/ObjectDetection.Pytorch/saved"
#WEIGHTS_SAVE_PATH="/data_host/trans/pytorch/OD.Pytorch_saved/saved"

#THRESHOLD=0.6
THRESHOLD=0.7
#THRESHOLD=0.8

THRESHOLD_P=`echo "$THRESHOLD * 100 / 1" | bc`
THRESHOLD_P=$THRESHOLD_P

DATASET_NAME="3sTFace_Manual_CVI"

#MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_resnet_bifpn.yaml"
#OUT_NAME="${DATASET_NAME}_resnet18-bifpn_final_8_thr70"
#RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0605/${DATASET_NAME}_ksevendet-resnet18-bifpn_final_8.pt"

MODEL_NAME="3sTFace_resnet18-bifpn-sh_ns-acr_x20_15"
MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_7.pt"
OUT_NAME="${MODEL_NAME}_ep-7_thr${THRESHOLD_P}"

INPUT_SHAPE="512,640"
python inference_result2video.py --dataset ${DATASET_NAME} \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE \
    --demo_path $DEMO_DIR \
    --threshold $THRESHOLD \
    --output_name $OUT_NAME

