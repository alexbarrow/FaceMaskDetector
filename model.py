from data_prep import FMDataset, get_transform, collate_fn, UnNormalize
import torch
import torchvision
from eval import CocoTypeEvaluator, get_coco_api_from_dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from logger import Logger, AverageMeter

import datetime
import time
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def prep_data(bs=6, tsr=0.2, seed=1):
    print('Datasets preparation...')
    train_transform, valid_transform = get_transform(train=True), get_transform(train=False)
    dataset = FMDataset('data/train/', train_transform)
    dataset_test = FMDataset('data/test/', valid_transform)

    # torch.manual_seed(seed)
    # indices = torch.randperm(len(dataset)).tolist()
    # tsize = int(len(dataset)*tsr)
    #
    # dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

    data_loader = DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    return data_loader, data_loader_test


def init_model(classes=4):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler=None):
    loss_total, loss_cls, loss_box, loss_obj, loss_rpn = AverageMeter(), AverageMeter(), AverageMeter(),\
                                                         AverageMeter(), AverageMeter()
    ev_time = time.time()
    model.train()

    for imgs, targs in data_loader:
        # unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # TODO: visualizing of image without boxes during train
        images = list(image.to(device) for image in imgs)
        batch_size = len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targs]
        loss_dict = model(images, targets)

        # print('target')
        # print(targets)
        # for key, img in zip(res.keys(), images):
        #     img = img.permute(1, 2, 0)
        #     visualize(unnorm(img), res[key]['boxes'].tolist(), res[key]['labels'].tolist())

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # TODO: check batch_size param, probably it should be iter (or epoch*batch_size)
        loss_total.update(losses.detach().item(), batch_size)
        loss_cls.update(loss_dict['loss_classifier'].detach().item(), batch_size)
        loss_box.update(loss_dict['loss_box_reg'].detach().item(), batch_size)
        loss_obj.update(loss_dict['loss_objectness'].detach().item(), batch_size)
        loss_rpn.update(loss_dict['loss_rpn_box_reg'].detach().item(), batch_size)

        if lr_scheduler is not None:
            lr_scheduler.step(metrics=losses)

    train_time = time.time() - ev_time

    return loss_total, loss_cls, loss_box, loss_obj, loss_rpn, optimizer.param_groups[0]["lr"], train_time


@torch.no_grad()
def evaluate(model, data_loader, device):
    # unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    print('Start evaluating...')
    evaluator_time = time.time()
    model.eval()

    coco_dataset = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoTypeEvaluator(coco_dataset)

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # for key, img in zip(res.keys(), images):
        #     img = img.permute(1, 2, 0)
        #     visualize(unnorm(img), res[key]['boxes'].tolist(), res[key]['labels'].tolist())

        coco_evaluator.update(res)

    print('EVALUATION TIME: (t={:0.2f}s)'.format(time.time() - evaluator_time))
    coco_evaluator.synchronize_between_processes()
    # print(coco_evaluator.coco_eval._paramsEval.imgIds)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator.coco_eval.stats


def main(num_epochs, no_log=False, lr_scheduler_val=True, save_checkpoint=False, save_fin_model=False):
    print('Object detection task on FMset.')
    check_pass = f'checkpoints/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(check_pass)
    model_name = 'frcnn_e{}_lr1-2_wd_1-4_ROP_p1_f2-1'.format(num_epochs)

    if no_log is False:
        logger = Logger(model_name)

    data_loader, data_loader_test = prep_data()
    print('Creating model...')
    model = init_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0001)

    lr_sc = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_sc = torch.optim.lr_scheduler.MultiStepLR(optimizer, [14, 19, 24, 28], 0.5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler_params = dict(
    #     mode='min',
    #     factor=0.5,
    #     patience=1,
    #     verbose=True,
    #     threshold=0.0001,
    #     threshold_mode='abs',
    #     cooldown=0,
    #     min_lr=1e-8,
    #     eps=1e-08
    # )
    # lr_sc = scheduler(optimizer, **scheduler_params)

    for epoch in range(num_epochs):
        print('Epoch: {}, Training...'.format(epoch+1))
        loss_total, loss_cls, loss_box, loss_obj, loss_rpn, lr, tr_time = \
            train_one_epoch(model, optimizer, data_loader, device=device, lr_scheduler=None)
        acc = evaluate(model, data_loader_test, device=device)

        if lr_scheduler_val is not False:
            # lr_sc.step(loss_total.avg)
            lr_sc.step()

        if no_log is False:
            logger.update_all({'loss_total': loss_total.avg, 'loss_cls': loss_cls.avg, 'loss_box': loss_box.avg,
                               'loss_obj': loss_obj.avg, 'loss_rpn': loss_rpn.avg,
                               'lr': lr, 'train_time': tr_time}, epoch+1)
            logger.update_acc(acc, epoch+1)
            logger.show_last()

        if save_checkpoint:
            if epoch > 10 and epoch % 5 == 0:
                model.eval()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_sc.state_dict(),
                    'epoch': epoch+1,
                }, check_pass +
                   f'/checkpoint-{str(epoch+1).zfill(3)}epoch.bin')

    if save_fin_model:
        torch.save(model.state_dict(), check_pass+f'/{model_name}-state_dict.pt')


if __name__ == '__main__':
    total_time = time.time()
    # main(6, no_log=False, lr_scheduler_val=True, save_fin_model=True, save_checkpoint=False)
    print(init_model())

    print('TOTAL TIME: (t={:0.2f}s)'.format(time.time() - total_time))