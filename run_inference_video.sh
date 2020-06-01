set -e 

OD_SAVED_PATH="/data_host/trans/pytorch/OD.Pytorch_saved/saved"
KSDET_MODEL_PATH="/data_host/trans/pytorch/ksevendet_models"
THERMAL_SAMPLE_PATH="/data_host/trans/client_thermal/sample_data"
INPUT_SHAPE="512,640"

#THRESHOLD=0.6
THRESHOLD=0.7
#THRESHOLD=0.8


MODEL_CONFIG="${KSDET_MODEL_PATH}/3sTFace_0515_Manual/mAP-608_resnet18_bifpn_final_4.yaml"
RESUME_PT="${KSDET_MODEL_PATH}/3sTFace_0515_Manual/mAP-608_resnet18_bifpn_final_4.pt"

#MODEL_CONFIG="config/model/ksevendet_shufflenetv2.yaml"
#OUT_NAME="clean_v1_shufflenetv2_100-fpn_4_thr75"
#RESUME_PT="saved/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-shufflenetv2_x1_0-fpn_final_4.pt"

#MODEL_CONFIG="config/model/ksevendet_mobilenetv2.yaml"
#OUT_NAME="clean_v1_mobilenetv2_100-fpn_4_thr80"
#RESUME_PT="saved/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-mobilenetv2_100-fpn_final_4.pt"

#MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
#OUT_NAME="manual_sample_resnet18-bifpn_4_thr70"
#RESUME_PT="/data_host/trans/pytorch/OD.Pytorch_saved/saved/final_weights_0/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-resnet18-bifpn_final_4.pt"

#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200521_DONE/CVI_TD_0521-000.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200521_DONE/CVI_TD_0521-001.avi"
INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-003.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-013.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-017.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-018.avi"
#INPUT_VIDEO_PATH="${THERMAL_SAMPLE_PATH}/Simple_Cut/CVI_TD_20200522_DONE/CVI_TD_0522-000.avi"

python inference_video.py \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --input_path $INPUT_VIDEO_PATH \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE \
    --threshold $THRESHOLD 
