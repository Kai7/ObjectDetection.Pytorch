import argparse
from pycocotools.cocoeval import COCOeval
from retinanet.dataloader import CocoDataset, FLIRDataset, CSVDataset
import json

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple evaluation script.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--dataset_root',
                        default='/root/data/VOCdevkit/',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--result_file',
                        default='result.json',
                        help='The result file of object detection.')
    parser = parser.parse_args(args)

    if parser.dataset == 'FLIR':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_val = FLIRDataset(parser.dataset_root, set_name='val')
        # print(dataset_val.flir.imgs)
    else:
        raise ValueError('Dataset type not understood (FLIR), exiting.')
    
    summarize = evaluate_flir(dataset_val, parser.result_file)
    # print(summarize)

def evaluate_flir(dataset, result_file, threshold=0.05):
    image_ids = []
    for index in range(len(dataset)):
        # data = dataset[index]
        # print(dataset.image_ids[index])
        image_ids.append(dataset.image_ids[index])

    # load results in FLIR evaluation tool
    flir_true = dataset.flir
    flir_pred = flir_true.loadRes(result_file)

    # run COCO evaluation
    flir_eval = COCOeval(flir_true, flir_pred, 'bbox')
    flir_eval.params.imgIds = image_ids
    flir_eval.evaluate()
    flir_eval.accumulate()
    flir_eval.summarize()

    stats_info = '''\
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.4f}
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:0.4f}
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:0.4f}
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.4f}
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.4f}
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.4f}
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.4f}\n'''
    
    return stats_info.format(*flir_eval.stats)

if __name__ == '__main__':
    main()