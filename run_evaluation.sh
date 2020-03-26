#COCO_DATA_ROOT='/data_host/dataset_zoo_tmp/coco/2017'
FLIR_DATA_ROOT='/data_host/FLIR_ADAS/FLIR_ADAS'

#python eval_FLIR.py --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --result val_bbox_results.json
echo " --- YoloV3 SPP Result"
python eval_FLIR.py --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --result yolov3_result_fixed.json

echo " --- RetinaNet Result"
python eval_FLIR.py --dataset FLIR --dataset_root "$FLIR_DATA_ROOT" --result val_bbox_results.json
