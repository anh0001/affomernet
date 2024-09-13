import numpy as np
from src.misc import dist

def voc_eval(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on VOC dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    class_num = len(PascalVOCDetection.class_names)
    pred_boxes = [boxlist.bbox.numpy() for boxlist in pred_boxlists]
    pred_labels = [boxlist.get_field("labels").numpy() for boxlist in pred_boxlists]
    pred_scores = [boxlist.get_field("scores").numpy() for boxlist in pred_boxlists]
    gt_boxes = [boxlist.bbox.numpy() for boxlist in gt_boxlists]
    gt_labels = [boxlist.get_field("labels").numpy() for boxlist in gt_boxlists]
    gt_difficults = [boxlist.get_field("difficult").numpy() if boxlist.has_field("difficult") else None
                     for boxlist in gt_boxlists]

    aps = []
    for class_index in range(class_num):
        rec, prec, ap = voc_eval_class(
            pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels, gt_difficults,
            class_index, iou_thresh, use_07_metric)
        aps.append(ap)

    return {"ap": aps, "map": np.mean(aps)}

def voc_eval_class(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                   gt_difficults=None, class_index=None, iou_thresh=0.5, use_07_metric=False):
    """Evaluate a certain class.
    """
    # ... [Implementation of voc_eval_class] ...
    # This function would contain the core evaluation logic, 
    # including matching predictions to ground truth, 
    # calculating precision and recall, and computing average precision.
    
    # For brevity, I'm not including the full implementation here, 
    # but it would follow the logic in the original voc_eval function.

    return rec, prec, ap

class PascalVOCEvaluator:
    def __init__(self, dataset, iou_thresh=0.5, use_07_metric=False):
        self.dataset = dataset
        self.iou_thresh = iou_thresh
        self.use_07_metric = use_07_metric
        self.reset()

    def reset(self):
        self.predictions = []
        self.gt_boxes = []

    def update(self, predictions):
        self.predictions.extend(predictions.values())

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = [item for sublist in all_predictions for item in sublist]

    def accumulate(self):
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            self.gt_boxes.append(target)

    def summarize(self):
        result = voc_eval(self.predictions, self.gt_boxes, self.iou_thresh, self.use_07_metric)
        
        print(f"mAP: {result['map']:.4f}")
        for i, ap in enumerate(result['ap']):
            print(f"{self.dataset.class_names[i]}: {ap:.4f}")

        return result