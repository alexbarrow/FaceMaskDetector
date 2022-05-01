import xml.etree.cElementTree as ET
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

T1_COLOR = (255, 0, 0)  # Red
T2_COLOR = (0, 0, 255)   # Blue
T3_COLOR = (0, 255, 0)   # Green
TEXT_COLOR = (0, 0, 0)  # White

id_to_labels = {1: 'with_mask', 2: 'without_mask', 3: 'mask_weared_incorrect'}


def get_bb_list(root_path):
    tree = ET.ElementTree(file=root_path)

    root = tree.getroot()
    bb_list = []
    class_labels = []

    for child_of_root in root.iter('object'):

        bb_temp = []
        for bb in child_of_root.iter('bndbox'):
            child = bb.getchildren()
            for ch in child:
                bb_temp.append(int(ch.text))

        for name in child_of_root.iter('name'):
            label = name.text
            class_labels.append(label)
        bb_list.append(bb_temp)

    return bb_list, class_labels


def get_img_size(root_path):
    tree = ET.ElementTree(file=root_path)
    root = tree.getroot()
    w, h = 0, 0

    for child in root.findall('size'):
        w = child.find('width').text
        h = child.find('height').text

    return int(w), int(h)


def visualize_bbox(img, bbox, class_name, color, score=False, sc_value=1, thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_DUPLEX, 0.35, 1)

    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    if not score:
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA
        )
    else:
        cv2.putText(
            img,
            text='{:0.2f}'.format(sc_value),
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img


def visualize(image, bboxes, labels):
    img = image.copy()
    for bbox, label in zip(bboxes, labels):
        if type(label) == int:
            class_name = id_to_labels[label]
        else:
            class_name = label

        if class_name == 'without_mask':
            color = T1_COLOR
        elif class_name == 'mask_weared_incorrect':
            color = T2_COLOR
        elif class_name == 'with_mask':
            color = T3_COLOR
        else:
            color = TEXT_COLOR

        img = visualize_bbox(img, bbox, class_name, color)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img.astype('uint8'))
    plt.show()


def visualize_inf(image, bboxes, labels, scores):
    img_dt = image.copy()
    for bbox, label, score in zip(bboxes, labels, scores):
        if type(label) == int:
            class_name = id_to_labels[label]
        else:
            class_name = label

        if class_name == 'without_mask':
            color = T1_COLOR
        elif class_name == 'mask_weared_incorrect':
            color = T2_COLOR
        elif class_name == 'with_mask':
            color = T3_COLOR
        else:
            color = TEXT_COLOR

        img_dt = visualize_bbox(img_dt, bbox, class_name, color, score=True, sc_value=score)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img_dt.astype('uint8'))
    plt.show()


def visualize_pair(img, real_pars, od_pars):
    img_dt = img.copy()
    img_real = img.copy()

    r_bbs, r_labels = real_pars
    bbs, labels, scores = od_pars

    for bbox, label, score in zip(bbs, labels, scores):
        if type(label) == int:
            class_name = id_to_labels[label]
        else:
            class_name = label

        if class_name == 'without_mask':
            color = T1_COLOR
        elif class_name == 'mask_weared_incorrect':
            color = T2_COLOR
        elif class_name == 'with_mask':
            color = T3_COLOR
        else:
            color = TEXT_COLOR

        img_dt = visualize_bbox(img_dt, bbox, class_name, color, score=True, sc_value=score)

    for bbox, label in zip(r_bbs, r_labels):
        if type(label) == int:
            class_name = id_to_labels[label]
        else:
            class_name = label

        if class_name == 'without_mask':
            color = T1_COLOR
        elif class_name == 'mask_weared_incorrect':
            color = T2_COLOR
        elif class_name == 'with_mask':
            color = T3_COLOR
        else:
            color = TEXT_COLOR

        img_real = visualize_bbox(img_real, bbox, class_name, color)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axs[0].imshow(img_real.astype('uint8'))
    axs[0].set_title('Real objects:', fontsize='x-large')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])

    axs[1].imshow(img_dt.astype('uint8'))
    axs[1].set_title('Detected objects:', fontsize='x-large')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])

    plt.show()


def get_sample(n=random.randint(0, 852), show_bb=True, fs=14, info=False, img_folder='data'):
    ann_path = img_folder + '/annotations/maksssksksss{}.xml'.format(n)
    img_path = img_folder + '/images/maksssksksss{}.png'.format(n)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not show_bb:
        plt.figure(figsize=(fs, fs))
        plt.imshow(img.astype('uint8'))
        plt.show()
    else:
        bb, class_labels = get_bb_list(ann_path)
        if info:
            w, h = get_img_size(ann_path)
            bb_area = (np.array(bb)[:, 3] - np.array(bb)[:, 1]) * (np.array(bb)[0:, 2] - np.array(bb)[0:, 0])
            print(bb, class_labels, bb_area/(w*h))
        visualize(img, bb, class_labels)


if __name__ == '__main__':
    get_sample(show_bb=True, info=True)