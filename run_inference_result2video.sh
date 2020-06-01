set -e 

#DEMO_DIR="/data_host/trans/client_thermal/train_valid/init_sample/train/images"
#DEMO_DIR="/data_host/trans/client_thermal/train_valid/sample_0/valid/images"
DEMO_DIR="/data_host/trans/client_thermal/train_valid/sample_manual_20200515/valid/images"
#THRESHOLD=0.6
THRESHOLD=0.7
#THRESHOLD=0.8


#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
#OUT_NAME="clean_v1_resnet18-fpn_4_thr80"
#RESUME_PT="saved/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-resnet18-fpn_final_4.pt"

#MODEL_CONFIG="config/model/ksevendet_shufflenetv2.yaml"
#OUT_NAME="clean_v1_shufflenetv2_100-fpn_4_thr75"
#RESUME_PT="saved/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-shufflenetv2_x1_0-fpn_final_4.pt"

#MODEL_CONFIG="config/model/ksevendet_mobilenetv2.yaml"
#OUT_NAME="clean_v1_mobilenetv2_100-fpn_4_thr80"
#RESUME_PT="saved/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-mobilenetv2_100-fpn_final_4.pt"

MODEL_CONFIG="config/model/ksevendet_resnet_bifpn.yaml"
OUT_NAME="manual_sample_resnet18-bifpn_4_thr70"
RESUME_PT="/data_host/trans/pytorch/OD.Pytorch_saved/saved/final_weights_0/3sTFaceSample_20200515_Manual_Clean_v1_ksevendet-resnet18-bifpn_final_4.pt"

INPUT_SHAPE="512,640"
python inference_result2video.py --dataset 3sTFaceSample_20200515_Manual_Clean_v1 \
    --model_config $MODEL_CONFIG \
    --resume $RESUME_PT \
    --num_classes 1 \
    --input_shape $INPUT_SHAPE \
    --demo_path $DEMO_DIR \
    --threshold $THRESHOLD \
    --output_name $OUT_NAME

#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
#INPUT_SHAPE="512,640"
#DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
#for i in {0..4};
#do
#    RESUME_PT="saved/final_weights/3sTFaceSample_${i}_ksevendet-resnet18-fpn_final_12.pt"
#    DEMO_DIR="${DATASET_ROOT_BASE}/sample_0511_clean/sample_${i}_clean/valid/images"
#    python inference_result2video.py --dataset 3sTFaceSample_${i} \
#        --model_config $MODEL_CONFIG \
#        --resume $RESUME_PT \
#        --num_classes 1 \
#        --input_shape $INPUT_SHAPE \
#        --demo_path $DEMO_DIR \
#        --threshold $THRESHOLD \
#        --output_name Sample_${i}_resnet18-fpn_final_12_thr09
#done
