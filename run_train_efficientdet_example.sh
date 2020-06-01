set -e 

DATASET_TYPE="kseven"

DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"
INPUT_SHAPE="512,640"

BATCH_SIZE=8
MODEL_CONFIG="config/model/ksevendet_resnet.yaml"
DATASET_ROOT="${DATASET_ROOT_BASE}/sample_manual_20200515"
python train_ksevendet.py --log --dataset 3sTFaceSample_20200515_Manual_Clean_v1 \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --architecture "efficientdet-d0" \
    --learning_rate 0.0001 \
    --weight_decay 0.00005 \
    --batch_size $BATCH_SIZE \
    --epochs 4 \
    --input_shape $INPUT_SHAPE \
    --valid_period 1

