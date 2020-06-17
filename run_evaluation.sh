set -e 

DATASET_TYPE="kseven"

## Thermal Face Training
DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
#DATASET_BASE_NAME="sample_manual_20200515_v1"
#DATASET_NAME="3sTFace_Manual_0515"
DATASET_BASE_NAME="sample_manual_CVI"
DATASET_NAME="3sTFace_Manual_CVI"

KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"

#INPUT_SHAPE="512,640"


#RESUME_PT="saved/final_weight_0527/3sTFace0515_Manual_ksevendet-resnet18-fpn_final_4.pt"
#    --resume $RESUME_PT \


##MODEL_CONFIG="${KSDET_MODEL_PATH}/3sTFace_0515_Manual/mAP-608_resnet18_bifpn_final_4.yaml"
##RESUME_PT="${KSDET_MODEL_PATH}/3sTFace_0515_Manual/mAP-608_resnet18_bifpn_final_4.pt"
##MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
##RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0604/3sTFace_Manual_CVI_ksevendet-resnet18-bifpn_final_8.pt"
#MODEL_CONFIG="config/model_3s-thermal-face/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0605/3sTFace_Manual_CVI_ksevendet-resnet18-bifpn_final_8.pt"
##RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0605/3sTFace_Manual_CVI_ksevendet-resnet18-bifpn_1.pt"
#DATASET_ROOT="${DATASET_ROOT_BASE}/${DATASET_BASE_NAME}"

#MODEL_NAME="3sTFace_320x384_resnet18-tfdbifpn-sh_ns-acr_x20_10"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-tfdbifpn_final_8.pt"
#INPUT_SHAPE="320,384"

####################################################################################################
#MODEL_NAME="3sTFace_512x640_resnet18-bifpn-sh_ns_3456-acr_x20_20"
#MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
#RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_final_8.pt"
#INPUT_SHAPE="512,640"
#        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.718
#        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.976
#        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.930
#        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717
#        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
####################################################################################################
MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
RESUME_PT="saved/${DATASET_NAME}_ksevendet-resnet18-bifpn_1.pt"
INPUT_SHAPE="512,640"
#        test 
####################################################################################################


DATASET_ROOT="${DATASET_ROOT_BASE}/${DATASET_BASE_NAME}"
python train_ksevendet.py --dataset ${DATASET_NAME} \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --input_shape $INPUT_SHAPE \
    --validation_only

