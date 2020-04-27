FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'
#THERMAL_DATA_ROOT='/data_host/trans/client_thermal/train_valid'
THERMAL_DATA_ROOT='/data_host/trans/client_thermal/valid'

RESUME_PATH="saved/thermal_RetinaNet-Tiny_4.pth"

python test.py --resume saved/weight/FLIR_RetinaNet-Res18_mAP074_14.pt --dataset thermal --dataset_root "$THERMAL_DATA_ROOT"

#python test.py --resume saved/weight/FLIR_RetinaNet-Res18_mAP074_14.pt --dataset FLIR --dataset_root "$FLIR_DATA_ROOT"
