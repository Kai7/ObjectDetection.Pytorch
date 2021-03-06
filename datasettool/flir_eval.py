from pycocotools.cocoeval import COCOeval
import json
import torch

import pdb

def evaluate_flir(dataset, model, threshold=0.05, valid_class_id=None):
    model.eval()
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = model(data['img'].permute(
                2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if valid_class_id and label not in valid_class_id:
                        continue

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_flir_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(
            dataset.set_name), 'w'), indent=4)

        # load results in FLIR evaluation tool
        flir_true = dataset.flir
        flir_pred = flir_true.loadRes(
            '{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        flir_eval = COCOeval(flir_true, flir_pred, 'bbox')
        flir_eval.params.imgIds = image_ids
        flir_eval.evaluate()
        flir_eval.accumulate()
        flir_eval.summarize()
        model.train()

        stats_info = '''\
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.3f}
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:0.3f}
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:0.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.3f}\n'''
        # print('tttt\n')
        # print(stats_info.format(*flir_eval.stats))
        return stats_info.format(*flir_eval.stats)