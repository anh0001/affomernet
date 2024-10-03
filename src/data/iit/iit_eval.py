import os
import numpy as np
import torch
from src.misc import dist
from collections import defaultdict
import tempfile
import pickle

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class IITEvaluator:
    def __init__(self, dataset, iou_thresh=0.5, use_07_metric=False):
        self.dataset = dataset
        self.iou_thresh = iou_thresh
        self.use_07_metric = use_07_metric
        self.class_names = dataset.class_names
        self.reset()

    def reset(self):
        self.image_ids = []
        self.bbox_predictions = defaultdict(list)
        self.mask_predictions = defaultdict(list)
        self.stats = np.zeros((2,), dtype=np.float64)  # Only two stats we store are mAP and AP@0.5

    def update(self, predictions):
        for image_id, prediction in predictions.items():
            image_id = str(image_id)
            self.image_ids.append(image_id)
            boxes = prediction['boxes'].tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()
            masks = prediction['masks'].cpu().numpy()

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                # Convert COCO format [x, y, width, height] to Pascal VOC format [xmin, ymin, xmax, ymax]
                xmin, ymin, w, h = box
                xmax, ymax = xmin + w, ymin + h
                self.bbox_predictions[self.class_names[label]].append([image_id, score, xmin, ymin, xmax, ymax])
                self.mask_predictions[self.class_names[label]].append([image_id, score, mask])

    def synchronize_between_processes(self):
        all_bbox_predictions = dist.all_gather(self.bbox_predictions)
        all_mask_predictions = dist.all_gather(self.mask_predictions)
        
        merged_bbox_predictions = defaultdict(list)
        merged_mask_predictions = defaultdict(list)
        
        for predictions in all_bbox_predictions:
            for key, value in predictions.items():
                merged_bbox_predictions[key].extend(value)
        
        for predictions in all_mask_predictions:
            for key, value in predictions.items():
                merged_mask_predictions[key].extend(value)
        
        self.bbox_predictions = merged_bbox_predictions
        self.mask_predictions = merged_mask_predictions

        all_image_ids = dist.all_gather(self.image_ids)
        self.image_ids = list(set([str(item) for sublist in all_image_ids for item in sublist]))

    def accumulate(self):
        # This method is not needed for this implementation
        pass

    def summarize(self):
        print('Summarizing results...')
        bbox_aps = []
        mask_aps = []

        # Create a temporary directory to store detection results
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, cls in enumerate(self.class_names):
                if cls == '__background__':
                    continue

                bbox_filename = os.path.join(tmp_dir, f'{cls}_bbox_det.txt')
                mask_filename = os.path.join(tmp_dir, f'{cls}_mask_det.txt')

                with open(bbox_filename, 'w') as f:
                    if cls in self.bbox_predictions:
                        for pred in self.bbox_predictions[cls]:
                            f.write(f"{pred[0]} {pred[1]} {pred[2]} {pred[3]} {pred[4]} {pred[5]}\n")

                with open(mask_filename, 'w') as f:
                    if cls in self.mask_predictions:
                        for pred in self.mask_predictions[cls]:
                            f.write(f"{pred[0]} {pred[1]} {pred[2]}\n")
                
                bbox_rec, bbox_prec, bbox_ap = self.voc_eval(
                    bbox_filename, 
                    self.dataset.annos_path,
                    self.dataset.imgs_path,
                    cls,
                    ovthresh=self.iou_thresh,
                    use_07_metric=self.use_07_metric,
                    use_mask=False
                )
                
                mask_rec, mask_prec, mask_ap = self.voc_eval(
                    mask_filename, 
                    self.dataset.annos_path,
                    self.dataset.imgs_path,
                    cls,
                    ovthresh=self.iou_thresh,
                    use_07_metric=self.use_07_metric,
                    use_mask=True
                )

                bbox_aps.append(bbox_ap)
                mask_aps.append(mask_ap)
                print(f'{cls}: BBox AP: {bbox_ap}, Mask AP: {mask_ap}')

        bbox_mAP = np.mean(bbox_aps)
        mask_mAP = np.mean(mask_aps)
        
        # Fill in the stats array with the calculated metrics
        self.stats[0] = bbox_mAP  # AP @ IoU=0.50:0.95 (we only calculated for IoU=0.5, so this is an approximation)
        self.stats[1] = bbox_mAP  # AP @ IoU=0.50
        self.stats[2] = mask_mAP
        self.stats[3] = mask_mAP
        # The rest of the stats are left as 0 since we haven't calculated them
        
        # Print summary (similar to COCO evaluator)
        print("BBox Average Precision (AP) @ [ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[0]))
        print("BBox Average Precision (AP) @ [ IoU=0.50      | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[1]))
        print("Mask Average Precision (AP) @ [ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[2]))
        print("Mask Average Precision (AP) @ [ IoU=0.50      | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[3]))

    def voc_eval(self, detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, use_mask=False):
        # First, load gt
        recs = {}
        for imagename in self.image_ids:
            recs[imagename] = self.parse_rec(os.path.join(annopath, f'{imagename}.xml'))

        # Extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in self.image_ids:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

        # Read dets
        with open(detpath, 'r') as f:
            lines = f.readlines()

        if use_mask:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            masks = [self.load_mask(x[0], classname) for x in splitlines]
        else:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        if use_mask:
            masks = [masks[x] for x in sorted_ind]
        else:
            BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # Go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            if use_mask:
                mask = masks[d]
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # Compute overlaps
                    for j in range(BBGT.shape[0]):
                        gt_mask = self.load_mask(image_ids[d], classname, j+1)
                        overlap = self.mask_iou(mask, gt_mask)
                        if overlap > ovmax:
                            ovmax = overlap
                            jmax = j
            else:
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # Compute overlaps
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # Union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # Compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def load_mask(self, image_id, classname, instance_id=None):
        mask_count = 1
        while True:
            mask_path = os.path.join(self.dataset.mask_cache_path, f'{image_id}_{mask_count}_segmask.sm')
            if not os.path.exists(mask_path):
                break
            with open(mask_path, 'rb') as f:
                mask = pickle.load(f)
            if instance_id is None or mask_count == instance_id:
                return (mask == self.class_names.index(classname)).astype(np.uint8)
            mask_count += 1
        return None

    def mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    @staticmethod
    def parse_rec(filename):
        return parse_rec(filename)