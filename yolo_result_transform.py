import argparse
import json
import pdb

IMAGE_ID_BIAS = -8863

def main(args=None):
    parser = argparse.ArgumentParser(description='Yolo result transform.')
    parser.add_argument('--result_file', default='yolov3_result.json',
                        help='The result file of object detection.')
    parser.add_argument('--out', default='yolov3_result_fixed.json',
                        help='The result fixed file of object detection.')
    parser = parser.parse_args(args)

    json_data = json.load(open(parser.result_file))
    for i in range(len(json_data)):
        ann = json_data[i]
        ann['image_id'] = ann['image_id'] + IMAGE_ID_BIAS
        ann['category_id'] = ann['category_id'] - 1
    # pdb.set_trace()

    with open(parser.out, 'w') as outfile: 
        json.dump(json_data, outfile, indent=4)

if __name__ == '__main__':
    main()