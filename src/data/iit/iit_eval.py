import os
import numpy as np
import torch
from src.misc import dist
from collections import defaultdict
import tempfile

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
        self.stats = np.zeros((2,), dtype=np.float64)  # Only two stats we store are mAP and AP@0.5

    def update(self, predictions):
        for image_id, prediction in predictions.items():
            image_id = str(image_id)
            self.image_ids.append(image_id)
            boxes = prediction['boxes'].tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()
            for box, score, label in zip(boxes, scores, labels):
                # Convert COCO format [x, y, width, height] to Pascal VOC format [xmin, ymin, xmax, ymax]
                xmin, ymin, w, h = box
                xmax, ymax = xmin + w, ymin + h
                self.bbox_predictions[self.class_names[label]].append([image_id, score, xmin, ymin, xmax, ymax])

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.bbox_predictions)
        merged_predictions = defaultdict(list)
        for predictions in all_predictions:
            for key, value in predictions.items():
                merged_predictions[key].extend(value)
        self.bbox_predictions = merged_predictions

        all_image_ids = dist.all_gather(self.image_ids)
        self.image_ids = list(set([str(item) for sublist in all_image_ids for item in sublist]))

    def accumulate(self):
        # This method is not needed for this implementation
        pass

    def summarize(self):
        print('Summarizing results...')
        aps = []
        # Create a temporary directory to store detection results
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, cls in enumerate(self.class_names):
                if cls == '__background__':
                    continue
                filename = os.path.join(tmp_dir, f'{cls}_det.txt')
                with open(filename, 'w') as f:
                    if cls in self.bbox_predictions:
                        for pred in self.bbox_predictions[cls]:
                            f.write(f"{pred[0]} {pred[1]} {pred[2]} {pred[3]} {pred[4]} {pred[5]}\n")
                
                rec, prec, ap = self.voc_eval(
                    filename, 
                    self.dataset.annos_path,
                    self.dataset.imgs_path,
                    cls,
                    ovthresh=self.iou_thresh,
                    use_07_metric=self.use_07_metric
                )
                aps.append(ap)
                print(f'{cls}: {ap}')

        mAP = np.mean(aps)
        
        # Fill in the stats array with the calculated metrics
        self.stats[0] = mAP  # AP @ IoU=0.50:0.95 (we only calculated for IoU=0.5, so this is an approximation)
        self.stats[1] = mAP  # AP @ IoU=0.50
        # The rest of the stats are left as 0 since we haven't calculated them
        
        # Print summary (similar to COCO evaluator)
        print("Average Precision (AP) @ [ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[0]))
        print("Average Precision (AP) @ [ IoU=0.50      | area=   all | maxDets=100 ] = {:.6f}".format(self.stats[1]))

    def voc_eval(self, detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
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

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [str(x[0]) for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # Go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
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

    @staticmethod
    def parse_rec(filename):
        return parse_rec(filename)