import os
import torch
import torch.utils.data
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import datapoints

from src.core import register

@register
class IITDetection(torch.utils.data.Dataset):
    __inject__ = ['transforms']
    
    class_names = ['__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife', 'cup', 'drill', 'racket', 'spatula', 'bottle']

    def __init__(self, root, year='2012', image_set='train', transforms=None, use_difficult=False):
        self.root = root
        self.year = year
        self.image_set = image_set
        self._transforms = transforms
        self.use_difficult = use_difficult
        
        self.imgs_path = os.path.join(self.root, 'JPEGImages')
        self.annos_path = os.path.join(self.root, 'Annotations')
        
        self._load_image_set_index()
        
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _load_image_set_index(self):
        image_set_file = os.path.join(self.root, 'ImageSets', 'Main', f'{self.image_set}.txt')
        with open(image_set_file) as f:
            self.ids = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.imgs_path, f'{img_id}.jpg')
        anno_path = os.path.join(self.annos_path, f'{img_id}.xml')
        
        img = Image.open(img_path).convert('RGB')
        target = self.parse_voc_xml(ET.parse(anno_path).getroot())
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, node):
        target = {}
        target['image_id'] = torch.tensor([self.ids.index(node.find('filename').text[:-4])])
        target['boxes'] = []
        target['labels'] = []
        target['area'] = []
        target['iscrowd'] = []
        target['difficult'] = []

        size = node.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for obj in node.findall('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.use_difficult and difficult:
                continue
            
            bbox = obj.find('bndbox')
            bbox = [float(bbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            target['boxes'].append(bbox)
            target['labels'].append(self.class_dict[obj.find('name').text.lower().strip()])
            target['area'].append(area)
            target['iscrowd'].append(0)
            target['difficult'].append(difficult)

        target['boxes'] = datapoints.BoundingBox(torch.tensor(target['boxes'], dtype=torch.float32), 
                                                 format=datapoints.BoundingBoxFormat.XYXY, 
                                                 spatial_size=(height, width))
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)
        target['difficult'] = torch.tensor(target['difficult'], dtype=torch.bool)
        target['orig_size'] = torch.as_tensor([int(node.find('size').find('width').text),
                                               int(node.find('size').find('height').text)])
        target['size'] = torch.as_tensor([int(node.find('size').find('width').text),
                                          int(node.find('size').find('height').text)])
        
        return target

    def extra_repr(self) -> str:
        return f'Split: {self.image_set}, Year: {self.year}'

iit_category2name = {i: name for i, name in enumerate(IITDetection.class_names)}
iit_category2label = {i: i for i in range(len(IITDetection.class_names))}
iit_label2category = {i: i for i in range(len(IITDetection.class_names))}