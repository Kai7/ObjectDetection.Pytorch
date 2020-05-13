set -e 

#DEMO_DIR="/data_host/trans/client_thermal/train_valid/init_sample/train/images"
DEMO_DIR="/data_host/trans/client_thermal/train_valid/sample_0/valid/images"
#THRESHOLD=0.6
THRESHOLD=0.9


#MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
#RESUME_PT="saved/3s-pocket-thermal-face_ksevendet-resnet18-fpn_final_12.pt"
#INPUT_SHAPE="512,640"
#python inference_result2video.py --dataset 3s-pocket-thermal-face \
#    --model_config $MODEL_CONFIG \
#    --resume $RESUME_PT \
#    --num_classes 1 \
#    --input_shape $INPUT_SHAPE \
#    --demo_path $DEMO_DIR \
#    --threshold $THRESHOLD \
#    --output_name resnet18-fpn_final_12_thr08

MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
INPUT_SHAPE="512,640"
DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
for i in {0..4};
do
    RESUME_PT="saved/final_weights/3sTFaceSample_${i}_ksevendet-resnet18-fpn_final_12.pt"
    DEMO_DIR="${DATASET_ROOT_BASE}/sample_0511_clean/sample_${i}_clean/valid/images"
    python inference_result2video.py --dataset 3sTFaceSample_${i} \
        --model_config $MODEL_CONFIG \
        --resume $RESUME_PT \
        --num_classes 1 \
        --input_shape $INPUT_SHAPE \
        --demo_path $DEMO_DIR \
        --threshold $THRESHOLD \
        --output_name Sample_${i}_resnet18-fpn_final_12_thr09
done
