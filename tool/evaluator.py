__all__= ['lht_CocoEvaluator', '_get_iou_types', 'lht_evaluate']

from .coco_eval import CocoEvaluator
from . utils import MetricLogger, SmoothedValue,reduce_dict
from . coco_utils import get_coco_api_from_dataset
import time
import random
import torch,tqdm
import torchvision
import numpy as np

#
class lht_CocoEvaluator(CocoEvaluator):
    def __init__(self, coco_gt, iou_types, **kwargs):
        super().__init__(coco_gt, iou_types)
        if 'areaRng' in kwargs:
            areaRng = kwargs['areaRng']
            for iou_type in iou_types:
                self.coco_eval[iou_type].params.areaRng = areaRng
#
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types
#
@torch.inference_mode()
def lht_evaluate(model, data_loader, device, areaRng):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter=" ")
    header = "Test:"
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    #areaRng = [[0, 10000000000.0], [0,225],[225,625],[625,10000000000.0]]
    coco_evaluator = lht_CocoEvaluator(coco, iou_types, areaRng=areaRng)
    mlog = metric_logger.log_every(data_loader, 100, header)
    if hasattr(tqdm, 'notebook'):
        pr_log = tqdm.notebook.tqdm(mlog,leave=True, total=len(data_loader))
    else:
        pr_log = tqdm.tqdm_notebook(mlog, total=len(data_loader), leave=True)
    for images, targets in pr_log:
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
#
@torch.inference_mode()
def lht_get_coco_evaluator(model, data_loader, device, areaRng):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = lht_CocoEvaluator(coco, iou_types, areaRng=areaRng)
    
    if hasattr(tqdm, 'notebook'):
        pr_log = tqdm.notebook.tqdm(data_loader,leave=False, total=len(data_loader))
    else:
        pr_log = tqdm.tqdm_notebook(data_loader,total=len(data_loader), leave=False)
    
    for images, targets in pr_log:
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        coco_evaluator.update(res)
        
        
    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    #coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
#
def _summarize(coco_eval,ap=1, iouThr=None, areaRng='all', maxDets=100 ):
    p = coco_eval.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap==1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    #print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s
#
def _summarizeDets(coco_eval):
    stats = np.zeros((12,))
    stats[0] = _summarize(coco_eval,ap=1)
    stats[1] = _summarize(coco_eval,ap=1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
    stats[2] = _summarize(coco_eval,ap=1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
    stats[3] = _summarize(coco_eval,ap=1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    stats[4] = _summarize(coco_eval,ap=1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    stats[5] = _summarize(coco_eval,ap=1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    stats[6] = _summarize(coco_eval,ap=0, maxDets=coco_eval.params.maxDets[0])
    stats[7] = _summarize(coco_eval,ap=0, maxDets=coco_eval.params.maxDets[1])
    stats[8] = _summarize(coco_eval,ap=0, maxDets=coco_eval.params.maxDets[2])
    stats[9] = _summarize(coco_eval,ap=0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    stats[10] = _summarize(coco_eval,ap=0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    stats[11] = _summarize(coco_eval,ap=0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    return stats
#
def lht_get_coco_stats(coco_evaluator, key='bbox'):
    coco_eval = coco_evaluator.coco_eval[key]
    stats = _summarizeDets(coco_eval)
    return stats


