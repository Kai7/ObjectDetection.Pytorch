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

from pycocotools.coco import COCO
if __name__ == '__main__':
    from pyflirtools import FLIR
else:
    from datasettool.pyflirtools import FLIR

from datasettool.cvidatatools import CVIData
from datasettool.ksevendata_tool import KSevenData

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

import pdb

class KSevenDataset(Dataset):
    """KSeven dataset."""
    def __init__(self, root_dir, set_name='train', annotation_name='annotations.json', transform=None):
        """
        Args:
            root_dir (string): KSeven directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.kseven_data = KSevenData(os.path.join(self.root_dir, self.set_name , annotation_name))
        print('Annotation File Path: ', os.path.join(self.root_dir, self.set_name , annotation_name))
        self.image_ids = self.kseven_data.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.kseven_data.loadCats(self.kseven_data.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.ksevendata_labels = {}
        self.ksevendata_labels_inverse = {}
        for c in categories:
            self.ksevendata_labels[len(self.classes)] = c['id']
            self.ksevendata_labels_inverse[c['id']] = len(self.classes)
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
        image_info = self.kseven_data.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # img = cv2.imread(path)
        # path = os.path.join(self.root_dir, 'images',
        #                     self.set_name, image_info['file_name'])
        # print('File Name:', image_info['file_name'])
        # print('Path:', path)
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0


    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.kseven_data.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        ksevendata_annotations = self.kseven_data.loadAnns(annotations_ids)
        for idx, a in enumerate(ksevendata_annotations):
            if a['category_id'] > 2:
                continue

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.ksevendata_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def ksevendata_label_to_label(self, ksevendata_label):
        return self.ksevendata_labels_inverse[ksevendata_label]

    def label_to_ksevendata_label(self, label):
        return self.ksevendata_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.kseven_data.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.kseven_data.dataset['categories'])


class CVIDataset(Dataset):
    """CVI dataset."""
    def __init__(self, root_dir, set_name='train', annotation_name='annotations.json', transform=None):
        """
        Args:
            root_dir (string): CVI directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.cvi_data = CVIData(os.path.join(self.root_dir, self.set_name , annotation_name))
        print('Annotation File Path: ', os.path.join(self.root_dir, self.set_name , annotation_name))
        self.image_ids = self.cvi_data.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.cvi_data.loadCats(self.cvi_data.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.cvidata_labels = {}
        self.cvidata_labels_inverse = {}
        for c in categories:
            self.cvidata_labels[len(self.classes)] = c['id']
            self.cvidata_labels_inverse[c['id']] = len(self.classes)
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
        image_info = self.cvi_data.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # img = cv2.imread(path)
        # path = os.path.join(self.root_dir, 'images',
        #                     self.set_name, image_info['file_name'])
        # print('File Name:', image_info['file_name'])
        # print('Path:', path)
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0


    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.cvi_data.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        cvidata_annotations = self.cvi_data.loadAnns(annotations_ids)
        for idx, a in enumerate(cvidata_annotations):
            if a['category_id'] > 2:
                continue

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.cvidata_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def cvidata_label_to_label(self, cvidata_label):
        return self.cvidata_labels_inverse[cvidata_label]

    def label_to_cvidata_label(self, label):
        return self.cvidata_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.cvi_data.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.cvi_data.dataset['categories'])

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
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
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        path       = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        #return 80
        return len(self.coco_labels)

class FLIRDataset(Dataset):
    """FLIR dataset."""

    def __init__(self, root_dir, set_name='train', annotation_name='thermal_annotations.json', transform=None):
        """
        Args:
            root_dir (string): FLIR directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.flir = FLIR(os.path.join(self.root_dir, self.set_name , annotation_name))
        print('Annotation File Path: ', os.path.join(self.root_dir, self.set_name , annotation_name))
        self.image_ids = self.flir.getImgIds()

        print(self.flir.dataset['categories'])
        # exit(0)

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
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # img = cv2.imread(path)
        # path = os.path.join(self.root_dir, 'images',
        #                     self.set_name, image_info['file_name'])
        # print('File Name:', image_info['file_name'])
        # print('Path:', path)
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0


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

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class K7ImagePreprocessor(object):
    def __init__(self, mean=None, std=None, resize_height=608, resize_width=1024, base_size=32, **kwargs):
        self.mean = mean if mean is not None else np.array([[[0.0, 0.0, 0.0]]])
        self.std  =  std if  std is not None else np.array([[[1.0, 1.0, 1.0]]])
        self.resize_height, self.resize_width = resize_height, resize_width
        self.base_size = base_size

        if 'logger' in kwargs:
            kwargs['logger'].info('Mean : {}'.format(str(self.mean)))
            kwargs['logger'].info('Std  : {}'.format(str(self.std)))
            kwargs['logger'].info('Resize Height  : {}'.format(self.resize_height))
            kwargs['logger'].info('Resize Width   : {}'.format(self.resize_width))
            kwargs['logger'].info('Base Size      : {}'.format(self.base_size))

    def __call__(self, sample):
        image = sample
        image = (image.astype(np.float32) - self.mean) / self.std

        rows, cols, cns = image.shape
        h_scale = self.resize_height / rows
        w_scale = self.resize_width  / cols 
        scale = min(h_scale, w_scale)
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        #pad_w = (self.base_size - rows%self.base_size)%self.base_size
        #pad_h = (self.base_size - cols%self.base_size)%self.base_size
        #new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)

        pad_h = (self.base_size - self.resize_height%self.base_size)%self.base_size
        pad_w = (self.base_size - self.resize_width%self.base_size)%self.base_size
        new_image = np.zeros((self.resize_height + pad_h, self.resize_width + pad_w, cns)).astype(np.float32)
        print(f'new_image.shape = {str(new_image.shape)}')

        new_image[:rows, :cols, :] = image.astype(np.float32)

        return new_image


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, height=608, width=1024, 
                    min_side=608, max_side=1024, base_size=32, 
                    inference_mode=False, resize_mode=0, **kwargs):
        self.height, self.width = height, width
        self.min_side, self.max_side = min_side, max_side
        self.base_size = base_size
        self.inference_mode = inference_mode
        self.resize_mode = resize_mode

        if 'logger' in kwargs:
            if self.resize_mode == 0:
                kwargs['logger'].info('Resizer.Min_side : {}'.format(self.min_side))
                kwargs['logger'].info('Resizer.Max_side : {}'.format(self.max_side))
            elif self.resize_mode == 1:
                kwargs['logger'].info('Resizer.Height  : {}'.format(self.height))
                kwargs['logger'].info('Resizer.Width   : {}'.format(self.width))

    def __call__(self, sample):
        if not self.inference_mode:
            image, annots = sample['img'], sample['annot']
        else: 
            image = sample
        rows, cols, cns = image.shape
        # print(image.shape)

        if self.resize_mode == 0:
            smallest_side = min(rows, cols)

            # rescale the image so the smallest side is min_side
            scale = self.min_side / smallest_side

            # check if the largest side is now greater than max_side, which can happen
            # when images have a large aspect ratio
            largest_side = max(rows, cols)

            if largest_side * scale > self.max_side:
                scale = self.max_side / largest_side

            # resize the image with the computed scale
            image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
            rows, cols, cns = image.shape

            pad_w = (self.base_size - rows%self.base_size)%self.base_size
            pad_h = (self.base_size - cols%self.base_size)%self.base_size

            new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
            new_image[:rows, :cols, :] = image.astype(np.float32)
            # print(new_image.shape)
        elif self.resize_mode == 1:
            h_scale = self.height / rows
            w_scale = self.width  / cols 
            scale = min(h_scale, w_scale)
            # print(f'resize scale = {scale}')

            # resize the image with the computed scale
            image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
            rows, cols, cns = image.shape

            #pad_w = (self.base_size - rows%self.base_size)%self.base_size
            #pad_h = (self.base_size - cols%self.base_size)%self.base_size

            pad_h = (self.base_size - self.height%self.base_size)%self.base_size
            pad_w = (self.base_size - self.width%self.base_size)%self.base_size

            new_image = np.zeros((self.height + pad_h, self.width + pad_w, cns)).astype(np.float32)
            # print(new_image.shape)
            new_image[:rows, :cols, :] = image.astype(np.float32)
        else:
            print('Error resize mode.')
            return None

        if not self.inference_mode:
            annots[:, :4] *= scale
            return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}
        else:
            return torch.from_numpy(new_image)


# class Resizer(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __init__(self, min_side=608, max_side=1024, base_size=32, inference_mode=False, **kwargs):
#         self.min_side  = min_side
#         self.max_side  = max_side
#         self.base_size = base_size
#         self.inference_mode = inference_mode

#         if 'logger' in kwargs:
#             kwargs['logger'].info('Resizer.Min_Side   : {}'.format(self.min_side))
#             kwargs['logger'].info('Resizer.Max_Side   : {}'.format(self.max_side))

#     def __call__(self, sample):
#         if not self.inference_mode:
#             image, annots = sample['img'], sample['annot']
#         else: 
#             image = sample
#         rows, cols, cns = image.shape
#         smallest_side = min(rows, cols)

#         # rescale the image so the smallest side is min_side
#         scale = self.min_side / smallest_side
#         print(f'scale = {scale}')

#         # check if the largest side is now greater than max_side, which can happen
#         # when images have a large aspect ratio
#         largest_side = max(rows, cols)

#         if largest_side * scale > self.max_side:
#             scale = self.max_side / largest_side

#         # resize the image with the computed scale
#         image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
#         rows, cols, cns = image.shape

#         #pad_w = self.base_size - rows%self.base_size
#         #pad_h = self.base_size - cols%self.base_size

#         pad_w = (self.base_size - rows%self.base_size)%self.base_size
#         pad_h = (self.base_size - cols%self.base_size)%self.base_size

#         new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
#         new_image[:rows, :cols, :] = image.astype(np.float32)
#         # print(new_image.shape)

#         if not self.inference_mode:
#             annots[:, :4] *= scale
#             return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}
#         else:
#             return torch.from_numpy(new_image)



class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, use_flip = True, flip_theta=0.5, 
                       use_noise = True, noise_theta=0.8, noise_range=2.0/255.0,
                       use_brightness = True, brightness_theta=0.2, brightness_range=0.05, 
                       use_scale = True, scale_theta=0.5, scale_max=1.0, scale_min=0.8, scale_stride=0.05,
                       **kwargs):
        self.use_flip = use_flip
        self.flip_theta = flip_theta

        self.use_noise = use_noise
        self.noise_theta = noise_theta
        self.noise_range = noise_range

        self.use_brightness = use_brightness
        self.brightness_theta = brightness_theta
        self.brightness_range = brightness_range

        self.use_scale = use_scale
        self.scale_theta = scale_theta
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.scale_stride = scale_stride
        self.scale_candidate = [scale_min,]
        s_min = scale_min
        while True:
            s_min += scale_stride
            if s_min <= scale_max:
                self.scale_candidate.append(s_min)
            else:
                break
        if scale_max not in self.scale_candidate:
            self.scale_candidate.append(scale_max)
        # print(self.scale_candidate)
        if 'logger' in kwargs:
            if self.use_flip:
                kwargs['logger'].info('Augmenter.Flip_Theta   : {}'.format(self.flip_theta))
            if self.use_noise:
                kwargs['logger'].info('Augmenter.Noise_Theta  : {}'.format(self.noise_theta))
                kwargs['logger'].info('Augmenter.Noise_Range  : {}'.format(self.noise_range))
            if self.use_brightness:
                kwargs['logger'].info('Augmenter.Brightness_Theta : {}'.format(self.brightness_theta))
                kwargs['logger'].info('Augmenter.Brightness_Range : {}'.format(self.brightness_range))
            if self.use_scale:
                kwargs['logger'].info('Augmenter.Scale_Theta  : {}'.format(self.scale_theta))
                kwargs['logger'].info('Augmenter.Scale_Max    : {}'.format(self.scale_max))
                kwargs['logger'].info('Augmenter.Scale_Min    : {}'.format(self.scale_min))
                kwargs['logger'].info('Augmenter.Scale_Stride : {}'.format(self.scale_stride))
                kwargs['logger'].info('Augmenter.Scale_Candidate : [ {} ] '.format(
                                       ', '.join(['{:.2f}'.format(s) for s in self.scale_candidate])))

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if self.use_flip and np.random.rand() < self.flip_theta:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

        if self.use_noise and np.random.rand() < self.noise_theta:
            row, col, ch = image.shape
            noise = np.random.uniform(-self.noise_range, self.noise_range, (row, col, 1))
            noise = np.concatenate((noise, noise, noise), axis=2)
            #image = image + np.random.uniform(-self.noise_range, self.noise_range, (row, col, ch))
            image = image + noise 
            image = np.clip(image, 0.0, 1.0)
            #gauss = np.random.normal(mean,sigma,(row,col,ch))
            #gauss = gauss.reshape(row,col,ch)

        if self.use_brightness and np.random.rand() < self.brightness_theta:
            image = image * np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            image = np.clip(image, 0.0, 1.0)
        
        if self.use_scale and np.random.rand() < self.scale_theta:
            random_scale = random.sample(self.scale_candidate, 1)[0]
            row, col, _ = image.shape
            image_scale = skimage.transform.resize(image, (int(round(row*random_scale)), int(round((col*random_scale)))))
            image = np.zeros((row, col, _)).astype(np.float32)
            row, col, _ = image_scale.shape
            image[:row, :col, :] = image_scale.astype(np.float32)

            annots = annots * random_scale

        sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, inference_mode=False):
        # ImageNet Setting
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std  = np.array([[[0.229, 0.224, 0.225]]])
        self.inference_mode = inference_mode

    def __call__(self, sample):
        if not self.inference_mode:
            image, annots = sample['img'], sample['annot']
            return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}
        else:
            image = sample
            return (image.astype(np.float32)-self.mean)/self.std

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
