set -e 

WEIGHTS_SAVE_PATH="/data_host/trans/pytorch/ObjectDetection.Pytorch/saved"
OD_SAVED_PATH="/data_host/trans/pytorch/OD.Pytorch_saved/saved"
KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"
THERMAL_SAMPLE_PATH="/data_host/trans/client_thermal/sample_data"
INPUT_SHAPE="512,640"

#THRESHOLD=0.6
#THRESHOLD=0.7
THRESHOLD=0.8
#THRESHOLD=0.9

THRESHOLD_P=`echo "$THRESHOLD * 100 / 1" | bc`
THRESHOLD_P=$THRESHOLD_P
#echo THRESHOLD_P=$THRESHOLD_P

DATASET_NAME="3sTFace_Manual_CVI"

#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"

#RESUME_PT="${WEIGHTS_SAVE_PATH}/final_weights_0604/${DATASET_NAME}_ksevendet-resnet18-fpn_final_8.pt"
#RESUME_PT="${KSDET_MODEL_PATH}/final_weights_0604/${DATASET_NAME}_ksevendet-resnet18-bifpn_4.pt"

#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200521_DONE/CVI_TD_0521-000.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200521_DONE/CVI_TD_0521-001.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-000.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-003.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-013.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-017.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-018.avi"

#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-010.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-011.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-012.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-013.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-014.avi"


MODEL_NAME="3sTFace_resnet18-bifpn-sh_ns-acr_x20_15"
MODEL_CONFIG="${KSDET_MODEL_PATH}/${MODEL_NAME}/ksevendet_resnet_bifpn.yaml"
RESUME_PT="${KSDET_MODEL_PATH}/${MODEL_NAME}/${DATASET_NAME}_ksevendet-resnet18-bifpn_7.pt"

#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-017.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-027.avi"
#python inference_video.py \
#    --model_config $MODEL_CONFIG \
#    --resume $RESUME_PT \
#    --input_path $INPUT_VIDEO_PATH \
#    --num_classes 1 \
#    --input_shape $INPUT_SHAPE \
#    --threshold $THRESHOLD 
for ii in {15..27}; do
    INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-0${ii}.avi"
    python inference_video.py \
        --model_config $MODEL_CONFIG \
        --resume $RESUME_PT \
        --input_path $INPUT_VIDEO_PATH \
        --num_classes 1 \
        --input_shape $INPUT_SHAPE \
        --model_name $MODEL_NAME \
        --threshold $THRESHOLD 
done

#for ii in {10..14}; do
#    INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200528_DONE/CVI_TD_0528-0${ii}.avi"
#    python inference_video.py \
#        --model_config $MODEL_CONFIG \
#        --resume $RESUME_PT \
#        --input_path $INPUT_VIDEO_PATH \
#        --num_classes 1 \
#        --input_shape $INPUT_SHAPE \
#        --model_name $MODEL_NAME \
#        --threshold $THRESHOLD 
#done
