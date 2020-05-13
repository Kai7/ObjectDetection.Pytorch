set -e 

DATASET_TYPE="kseven"

DATASET_ROOT_BASE="/data_host/trans/client_thermal/train_valid"

INPUT_SHAPE="512,640"
BATCH_SIZE=16
i=0
DATASET_ROOT="${DATASET_ROOT_BASE}/sample_0511_clean/sample_${i}_clean"
python unit_test_augmentation.py --dataset 3sTFaceSample_${i} \
    --dataset_root $DATASET_ROOT \
    --dataset_type $DATASET_TYPE \
    --batch_size $BATCH_SIZE \
    --input_shape $INPUT_SHAPE \

