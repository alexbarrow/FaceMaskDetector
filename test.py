from image_handler import visualize, visualize_pair, get_bb_list
from data_prep import UnNormalize, prep_to_prediction
import pandas as pd
import numpy as np
import os
import cv2
from model import init_model
import torch


def test_function(model_path, num_img, seed=1):
    rseed = seed
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = init_model()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    np.random.seed(rseed)

    names = np.array((sorted(os.listdir('data/test/images/'))))
    index = np.random.choice(names.shape[0], num_img, replace=False)

    for i in index:
        ann_path = 'data/test/annotations/' + ''.join(list(str(names[i]))[:-3]) + 'xml'
        img_path = 'data/test/images/' + names[i]

        real_bb, real_class_labels = get_bb_list(ann_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_to_dt, real_bb = prep_to_prediction(img, real_bb, real_class_labels)
        img_list = [img_to_dt.to(device)]

        outputs = model(img_list)

        boxes, labels, scores = outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']
        df = pd.DataFrame({'boxes': boxes.tolist(), 'labels': labels.tolist(), 'scores': scores.tolist()},
                          dtype='object')
        cr_df = df.loc[df['scores'] > 0.9]
        unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = img_to_dt.permute(1, 2, 0)

        visualize_pair(unnorm(img), (real_bb, real_class_labels), (cr_df['boxes'].tolist(), cr_df['labels'].tolist(),
                                                                   cr_df['scores'].tolist()))

        print('Name of image:  {}'.format(names[i]))


if __name__ == '__main__':
    model_path = 'checkpoints/20211118-175215/frcnn_e30_lr1-2_wd_1-4_ROP_p1_f2-1-state_dict.pt'
    seed = 1
    test_function(model_path, 10, seed=seed)
