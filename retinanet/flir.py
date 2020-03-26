from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO
if __name__ == '__main__':
    from pyflirtools import FLIR
else:
    from datasets.pyflirtools import FLIR

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image


class FLIRDataset(Dataset):
    """FLIR dataset."""

    def __init__(self, root_dir, set_name='train', transform=None):
        """
        Args:
            root_dir (string): FLIR directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.flir = FLIR(os.path.join(self.root_dir, self.set_name , 'thermal_annotations.json'))
        print('annotation_file: ', os.path.join(self.root_dir, self.set_name , 'thermal_annotations.json'))
        self.image_ids = self.flir.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.flir.loadCats(self.flir.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.flir_labels = {}
        self.flir_labels_inverse = {}
        for c in categories:
            self.flir_labels[len(self.classes)] = c['id']
            self.flir_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        # print('image')
        # print(img)
        # print('annotation')
        # print(annot)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.flir.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, 'images',
        #                     self.set_name, image_info['file_name'])
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # print('File Name:', image_info['file_name'])
        # print('Path:', path)
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.flir.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        flir_annotations = self.flir.loadAnns(annotations_ids)
        for idx, a in enumerate(flir_annotations):
            if a['category_id'] > 2:
                continue

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.flir_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def flir_label_to_label(self, flir_label):
        return self.flir_labels_inverse[flir_label]

    def label_to_flir_label(self, label):
        return self.flir_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.flir.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 3



if __name__ == '__main__':
    from augmentation import get_augumentation, Resizer, Normalizer, Augmenter
    # from augmentation import get_augumentation
    # dataset = FLIRDataset(root_dir='/data_host/FLIR_ADAS/FLIR_ADAS_1_3', set_name='train',
    #                       transform=get_augumentation(phase='train'))
    # dataset = FLIRDataset(root_dir='/data_host/FLIR_ADAS/FLIR_ADAS', set_name='train',
    #                       transform=transforms.Compose([Normalizer(),Augmenter(),Resizer()]))
    dataset = FLIRDataset(root_dir='/data_host/FLIR_ADAS/FLIR_ADAS', set_name='val',
                          transform=transforms.Compose([Normalizer(),Augmenter(),Resizer()]))
    
    # rand_id = 0
    rand_id = random.randint(0,len(dataset) - 1)
    sample = dataset[rand_id]
    
    # print('sample: ', sample)
    dataset.flir.info()
    img = sample['img'].numpy()
    annot = sample['annot'].numpy()
    print('img:')
    print(img)
    print(img.shape)
    print('annot:')
    print(annot)
    print('len(dataset) = ', len(dataset))
    print(dataset.classes)

    print('ID:', rand_id)

    import pdb
    pdb.set_trace()