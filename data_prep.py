import os
import cv2
import torch
import numpy as np
import shutil

from torch.utils.data import DataLoader

from albumentations.pytorch import ToTensorV2

from image_handler import get_bb_list, visualize
import albumentations as A


def collate_fn(batch):
    return tuple(zip(*batch))


def prep_to_prediction(img, bb, lbls):
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(height=350, width=350, p=1.0),
        ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['class_labels'])
    )

    transformed = transform(image=img, bboxes=bb, class_labels=lbls)
    img = transformed['image']
    bb = transformed['bboxes']
    return img, bb


def data_split(data_path, train_path, test_path, split_size=0.2, rseed=42):
    np.random.seed(rseed)
    names = np.array((sorted(os.listdir(os.path.join(data_path, "images")))))
    index = np.random.choice(names.shape[0], round(names.shape[0]*split_size), replace=False)
    train = np.setdiff1d(names, names[index])

    for file in names[index]:
        shutil.copy(data_path+"images/"+str(file), test_path+"images/")
        shutil.copy(data_path + "annotations/" + ''.join(list(str(file))[:-3])+'xml',
                    test_path + "annotations/")

    for file in train:
        shutil.copy(data_path+"images/"+str(file), train_path+"images/")
        shutil.copy(data_path + "annotations/" + ''.join(list(str(file))[:-3])+'xml',
                    train_path + "annotations/")
    print('Ready')


labels_to_id = {'with_mask': 1, 'without_mask': 2, 'mask_weared_incorrect': 3}


class FMDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images/", self.imgs[idx])
        box_path = os.path.join(self.root, "annotations/", self.boxes[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb, class_labels = get_bb_list(box_path)
        class_labels = [labels_to_id[label] for label in class_labels]

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bb, class_labels=class_labels)
            img = transformed['image']
            bb = transformed['bboxes']
            class_labels = transformed['class_labels']

        # TRANSFORM TO TENSORS

        class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
        bb = torch.as_tensor(bb, dtype=torch.float32)
        image_id = torch.tensor([idx])

        area = (bb[:, 3] - bb[:, 1]) * (bb[:, 2] - bb[:, 0])

        iscrowd = torch.zeros_like(area, dtype=torch.int64)

        target = {
            "boxes": bb,
            "labels": class_labels,
            "image_id": image_id,
            "area": area,
            'iscrowd': iscrowd
        }

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    if train:
        transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=350, height=350, p=1),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Resize(height=350, width=350, p=1.0),
            ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['class_labels']))
        return transform
    else:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize(height=350, width=350, p=1.0),
            ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0,
                                     label_fields=['class_labels']))

        return transform


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)*255.0
        self.std = np.array(std, dtype=np.float32)*255.0

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        img = tensor.cpu().numpy()
        img *= self.std
        img += self.mean

        return img


if __name__ == '__main__':
    # data_split('data/', 'data/train/', 'data/test/')
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = get_transform(train=True)
    dataset = FMDataset('data/', transform)

    # img = image_0.permute(1, 2, 0)
    # visualize(unnorm(img), target_0['boxes'], target_0['labels'].tolist())

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    # coco_dataset = get_coco_api_from_dataset(data_loader.dataset)
    # coco_evaluator = CocoTypeEvaluator(coco_dataset)

    for image, target in data_loader:
        images = list(image for image in image)
        targets = [{k: v for k, v in t.items()} for t in target]
    #
        print(targets)
    #
        image_0, target_0 = images[0], targets[0]
        image_0 = image_0.permute(1, 2, 0)

        visualize(unnorm(image_0), target_0['boxes'], target['labels'].tolist())
