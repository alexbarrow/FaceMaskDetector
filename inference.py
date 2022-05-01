from image_handler import visualize_inf
from data_prep import UnNormalize
import pandas as pd
import numpy as np
import os
import cv2
from model import init_model
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def od_inference(inf_path, model_path, threshold=0.65):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = init_model()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2()])

    names = np.array((sorted(os.listdir(inf_path))))

    for name in names:
        path = files_path + name

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_dt = img.copy()
        transformed = transform(image=img_dt)
        img_dt = transformed['image']

        img_list = [img_dt.to(device)]
        outputs = model(img_list)

        boxes, labels, scores = outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']

        # print(scores)
        df = pd.DataFrame({'boxes': boxes.tolist(), 'labels': labels.tolist(), 'scores': scores.tolist()}, dtype='object')
        cr_df = df.loc[df['scores'] > threshold]

        unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = img_dt.permute(1, 2, 0)

        visualize_inf(unnorm(img), cr_df['boxes'].tolist(), cr_df['labels'].tolist(), cr_df['scores'].tolist())
        print(name)


if __name__ == '__main__':
    files_path = 'data/inference/'
    model_path = 'checkpoints/20211118-175215/frcnn_e30_lr1-2_wd_1-4_ROP_p1_f2-1-state_dict.pt'
    od_inference(files_path, model_path, threshold=0.65)
