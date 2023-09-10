__all__ =['get_transform','BBoxDataset','collate_fn','FasterRCNN_VGG16_BN','lht_ResNetFPNBackBoneByLayerID','lhtInitializeFasterRCNN_FPN_ResNet',
          'lht_get_cuda_info','constructModel','lht_train_one_epoch','lht_getSmallMediumLargeDS','getAPSmallMediumBig',
         ]

import torch
import numpy as np
import os,sys,math,tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.models.resnet as resnet
import torchvision.transforms as T
from .utils import MetricLogger, SmoothedValue,reduce_dict
from . import evaluator as lhtevaluator

#
resnet_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
resnet_weights= { 'resnet18':resnet.ResNet18_Weights.IMAGENET1K_V1,
                  'resnet34':resnet.ResNet34_Weights.IMAGENET1K_V1,
                  'resnet50':resnet.ResNet50_Weights.IMAGENET1K_V1,
                  'resnet101':resnet.ResNet101_Weights.IMAGENET1K_V1,
                  'resnet152':resnet.ResNet152_Weights.IMAGENET1K_V1,
                }

def collate_fn(batch):
    return tuple(zip(*batch))
def get_transform(train=False):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
#
class BBoxDataset(torch.utils.data.Dataset):
    def __init__(self, root,pid=0,imgs=None, anns=None,transform=None):
        self.root = root
        self.pid = str(pid)
        self.transform = transform
        if imgs is None: 
            self.imgs = list(sorted(os.listdir(os.path.join(root, "JPGImages",self.pid))))
        else:
            self.imgs = imgs
        if anns is None:
            self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        else:
            self.annotations = anns
    def __getitem__(self, idx):
        img_path = os.path.join(self.root,"JPGImages",self.pid,self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(self.root,"Annotations", self.annotations[idx])
        ann = np.loadtxt(ann_path)
        if ann.ndim == 1:
            ann = ann[None]
        #
        category_id = np.int64(ann[:,0])
        category_id = torch.as_tensor(category_id, dtype=torch.int64)
        boxes = ann[:,1:]
        num_objs = boxes.shape[0]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['category_id'] = category_id
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.imgs)
#
def FasterRCNN_VGG16_BN(weights=None,rpn_sizes=((128, 256, 512),), rpn_aspect_ratios=((0.5, 1.0, 2.0),),
                        roi_featmap_names=['0'],roi_output_size=(7,7),num_classes=2):
    backbone = torchvision.models.vgg16_bn(weights=weights).features
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=rpn_sizes, aspect_ratios=rpn_aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names= roi_featmap_names,
                                                output_size= roi_output_size,
                                                sampling_ratio=2)
    #
    out_channels=512
    resolution = roi_output_size[0] * roi_output_size[1]
    representation_size=1024
    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(out_channels*resolution, representation_size)
    #
    model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler, box_head=box_head,)
    return model
#
def lht_ResNetFPNBackBoneByLayerID(resnet_name,weights=None,trainable_layers=3,returned_layers=[1,2,3,4]):
    '''construct backbone based on different layers from resnet
    Parameters:
        resnet_name:[str] ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        weights: torchvision.models.resnet.ResNetXX_Weights.DEFAULT. XX=18,34,50,101,or 152
        trainable_layers:[int] 0~5 -> layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        returned_layer:[int] [1,2,3,4] if it is 1, output of layer1 is returned
    '''
    bb = resnet_fpn_backbone(backbone_name=resnet_name, weights=weights, 
                             trainable_layers=trainable_layers, returned_layers=returned_layers)
    return bb
#
def lhtInitializeFasterRCNN_FPN_ResNet(rpn_anchor_generator, roi_output_size, featmap_names, bb_res_fpn, 
                                       num_classes=2,mean_train=None, std_train=None ,**kwargs):
    output_size = roi_output_size
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size,sampling_ratio=-1,
                                       canonical_scale=640,canonical_level=4)
    #construct box_head by output_size
    out_channels = bb_res_fpn.out_channels
    resolution = output_size[0] * output_size[1]
    representation_size=1024
    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(out_channels*resolution, representation_size)
    frcnn_fpn = torchvision.models.detection.FasterRCNN(backbone=bb_res_fpn,num_classes=num_classes,min_size=512,max_size=640,box_nms_thresh=0.5,
                                                 image_mean=mean_train,image_std=std_train,
                                                 rpn_anchor_generator=rpn_anchor_generator,
                                                 box_roi_pool=roi_pooler,box_head=box_head,
                                                )
    return frcnn_fpn
#
def constructModel(name='M1', num_classes = 2):
    if name == 'M2':
        return FasterRCNN_VGG16_BN()
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    featmap_names=['0','1','2','3']
    bb_res_fpn = lht_ResNetFPNBackBoneByLayerID(resnet_name='resnet152',weights=None,trainable_layers=5,returned_layers=[1,2,3,4])
    if name == 'M1':
        anchor_sizes = ((4,8,16,32,64),)*5
        roi_output_size=(8,10)
        rpn_anchor_generator =  AnchorGenerator(anchor_sizes, aspect_ratios)
        m = lhtInitializeFasterRCNN_FPN_ResNet(rpn_anchor_generator=rpn_anchor_generator, roi_output_size=roi_output_size,
                                                         featmap_names=featmap_names, bb_res_fpn=bb_res_fpn, num_classes=num_classes)
        return m
    if name == 'M3':
        anchor_sizes = ((128,256,512),)*5
        roi_output_size=(8,10)
        rpn_anchor_generator =  AnchorGenerator(anchor_sizes, aspect_ratios)
        m = lhtInitializeFasterRCNN_FPN_ResNet(rpn_anchor_generator=rpn_anchor_generator, roi_output_size=roi_output_size,
                                                         featmap_names=featmap_names, bb_res_fpn=bb_res_fpn, num_classes=num_classes)
        return m
    if name == 'M4':
        anchor_sizes = ((4,8,16,32,64),)*5
        roi_output_size=(7,7)
        rpn_anchor_generator =  AnchorGenerator(anchor_sizes, aspect_ratios)
        m = lhtInitializeFasterRCNN_FPN_ResNet(rpn_anchor_generator=rpn_anchor_generator, roi_output_size=roi_output_size,
                                                         featmap_names=featmap_names, bb_res_fpn=bb_res_fpn, num_classes=num_classes)
        return m
#
def lht_get_cuda_info():
    flag = torch.cuda.is_available()
    if not flag:
        print("No GPU")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        out = f"id: {i}, name: {torch.cuda.get_device_name(i)}"
        print(out)
        print("Memory Usage")
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
        print('Max memory reserved: ', round(torch.cuda.max_memory_reserved(i)/1024**3,1), 'GB')
        print('Total memory: ', round(torch.cuda.get_device_properties(i).total_memory/1024**3,1), 'GB')
        ret = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        print('Free memory: ', round(ret/1024**3,1), 'GB')
        print("-" * 20)
#
def lht_train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq,scaler=None,warm=False):
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None
    if epoch==0 and warm:
        warmup_factor = 1.0
        end_factor    =0.01
        warmup_iters = min(1000, len(data_loader)-1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=warmup_factor,end_factor=end_factor,total_iters=warmup_iters)
    #
    mlog = metric_logger.log_every(data_loader, print_freq, header)
    if hasattr(tqdm, 'notebook'):
        mlog_iter = tqdm.notebook.tqdm(mlog,total=len(data_loader), leave=False)
    else:
        mlog_iter = tqdm.tqdm_notebook(mlog,total=len(data_loader), leave=False)
    for images, targets in mlog_iter:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images,targets)
            losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        #
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        #
        metric_logger.update(loss=losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

#
def lht_getSmallMediumLargeDS(testAnns, testImgs,root,rng=[187,408]):
    ann_path = os.path.join(root,"Annotations")
    ids_large, ids_m, ids_s = [],[],[]
    fun_area = lambda x: (x[2]-x[0])*(x[3]-x[1])
    amin, amax = rng
    if hasattr(tqdm,'notebook'):
        pr = tqdm.notebook.tqdm(testAnns, leave=False)
    else:
        pr = tqdm.tqdm_notebook(testAnns, leave=False)
    for tid, a in enumerate(pr):
        ap = os.path.join(ann_path,a)
        ann = np.loadtxt(ap)
        if ann.ndim < 2:
            ann = ann[None]
        s = [fun_area(i) for i in ann[:,1:]]
        s = np.array(s)
        r = np.sum(s>amax)/len(s)
        if r >0.5:
            ids_large.append(tid)
            continue
        con = np.logical_and(s>amin, s<=amax)
        r = np.sum(con)/len(s)
        if r > 0.5:
            ids_m.append(tid)
            continue

        ids_s.append(tid)
    #
    # small
    testImgs_s = [testImgs[i] for i in ids_s]
    testAnns_s = [testAnns[i] for i in ids_s]
    # medium
    testImgs_m = [testImgs[i] for i in ids_m]
    testAnns_m = [testAnns[i] for i in ids_m]
    # large
    testImgs_b = [testImgs[i] for i in ids_large]
    testAnns_b = [testAnns[i] for i in ids_large]
    #
    testDS  = BBoxDataset(root=root,imgs=testImgs,anns=testAnns, transform=get_transform())
    #
    testDS_s  = BBoxDataset(root=root,imgs=testImgs_s,anns=testAnns_s, transform=get_transform())
    testDS_m  = BBoxDataset(root=root,imgs=testImgs_m,anns=testAnns_m, transform=get_transform())
    testDS_b  = BBoxDataset(root=root,imgs=testImgs_b,anns=testAnns_b, transform=get_transform())
    #
    test_loader_s  = torch.utils.data.DataLoader(testDS_s,batch_size=5,shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader_m  = torch.utils.data.DataLoader(testDS_m,batch_size=5,shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader_b  = torch.utils.data.DataLoader(testDS_b,batch_size=5,shuffle=False, num_workers=2, collate_fn=collate_fn)
    #
    test_loader  = torch.utils.data.DataLoader(testDS,batch_size=5,shuffle=False, num_workers=2, collate_fn=collate_fn)
    #
    return {'small':test_loader_s,'medium':test_loader_m,'big':test_loader_b,'all':test_loader}

#
def getAPSmallMediumBig(model,testAnns, testImgs,root,areaRng,rng=[187,408]):
    device = torch.device("cuda")
    model.to(device)
    smb = lht_getSmallMediumLargeDS(testAnns, testImgs,root,rng)
    ret = {}
    for n,ld in smb.items():
        coco_evaluator = lhtevaluator.lht_get_coco_evaluator(model,ld,device=device,areaRng=areaRng)
        s = lhtevaluator.lht_get_coco_stats(coco_evaluator)
        if n == 'all':
            ret[n] = [s[0],s[1],s[3],s[4],s[5],s[9],s[10],s[11]]
        else:
            ret[n] = s[1]
    #
    ap_s,ap_m,ap_b = ret['small'],ret['medium'],ret['big']
    ap_all = ret['all'][1] 
    ap_all_95 = ret['all'][0] 
    ap_all_95_s = ret['all'][2]
    ap_all_95_m = ret['all'][3]
    ap_all_95_b = ret['all'][4]
    #
    ar_all_95_s = ret['all'][5]
    ar_all_95_m = ret['all'][6]
    ar_all_95_b = ret['all'][7]
    
    return {'AP_0.5_all':ap_all, 'AP_0.5_small':ap_s,'AP_0.5_medium':ap_m,'AP_0.5_large':ap_b,
            'AP_0.5:0.95_small':ap_all_95_s,'AP_0.5:0.95_medium':ap_all_95_m,'AP_0.5:0.95_large':ap_all_95_b,
            'AR_0.5:0.95_small':ar_all_95_s,'AR_0.5:0.95_medium':ar_all_95_m,'AR_0.5:0.95_large':ar_all_95_b,
           }
#
def lht_vis_img(img, ax=None, figsize=(10,8)):
    if img.shape[-1] != 3:
        img = img.transpose(1,2,0)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(1,1,1)
    #
    ax.imshow(img.astype(np.uint8))
    return ax
#
def lht_vis_bbox(img, bbox, score=None, ax=None, label_names=None, label=None, edgecolor='red',minarea=0,figsize=(5,4)):
    if img.shape[-1] != 3:
        img = img.transpose(1,2,0)
    if ax is None:
        ax = lht_vis_img(img,ax=None, figsize=figsize)
    else:
        ax.imshow(img.astype(np.uint8))
    if len(bbox) == 0:
        return ax
    for i, bb in enumerate(bbox):
        y,x = (bb[1], bb[0])
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if w*h < minarea:
            continue
        rect = patches.Rectangle((x,y),w,h,linewidth=2, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
        caption = []
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
            pass
        if score is not None:
            sc = score[i]
            caption.append("{:.4f}".format(sc))
        if len(caption) > 0:
            ax.text(x,y, ": ".join(caption), style='italic', c='blue',
                    bbox={'facecolor': 'white', 'alpha':0.3, 'pad':0}
                   )
    return ax
    pass

def lht_read_image(path, dtype=np.float32, color="ttt", transform=None):
    f = Image.open(path)
    try:
        if color == "RGB":
            img = f.convert('RGB')
        elif color == "P":
            img = f.convert('P')
        else:
            img = f
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    if transform is None:
        return img
    else:
        return transform(img)

#
def smallObjDetection(model,img_path):
    img = Image.open(img_path)
    img = get_transform()(img)
    model.cpu()
    model.eval()
    preds = model([img])
    return img, preds[0]

#
@torch.inference_mode()
def lht_predictByDSLoader(dsLoader, device, model):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    preds = []
    if hasattr(tqdm, 'notebook'):
        pr_iter = tqdm.notebook.tqdm(dsLoader,total=len(dsLoader), leave=False)
    else:
        pr_iter = tqdm.tqdm_notebook(dsLoader,total=len(dsLoader), leave=False)
    for imgs,tgts in pr_iter:
        imgs = list(img.to(device) for img in imgs) 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        p = model(imgs)
        preds += p
    #
    torch.set_num_threads(n_threads)
    return preds