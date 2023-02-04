import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

data_dir = "/content/invoices"

data_files = []
labels = []

for root, dirs, files in os.walk(os.path.abspath(data_dir)):
    for base_name in files:
        image_file_path = os.path.join(root, base_name)

        if base_name.endswith(".jpg"):

            annotation_file_path = os.path.join(root.replace("/images/", "/labels/"), base_name.split(".jpg")[0] + ".txt")

            data_files.append([image_file_path, annotation_file_path])

            category_ids = [0]*19
            with open(annotation_file_path) as txtf:
                for line in txtf.readlines():
                    objects = line.split(" ")
                    category_ids[int(objects[0])] = 1

            labels.append(category_ids)

data_files = np.array(data_files)
labels = np.array(labels)

from skmultilearn.model_selection import iterative_train_test_split
train_files, train_labels, test_files, test_labels = iterative_train_test_split(data_files, labels, test_size = 0.2)

def make_npdata(data_files):
    
    images = []
    bboxes = []
    classes_ids = []

    for data_file in tqdm(data_files, position=0, leave=True):
        image_file, annotation_file = data_file[0], data_file[1]
        image = tf.io.encode_jpeg(tf.io.decode_jpeg(tf.io.read_file(image_file))).numpy()

        class_ids = []
        bboxs = []
        with open(annotation_file) as txtf:
            for line in txtf.readlines():
                objects = line.split(" ")
                class_ids.append(int(objects[0]))

                h= float(objects[4].rstrip("\n"))
                w = float(objects[3])
                x = float(objects[1])
                y = float(objects[2])
                x1 = float(objects[1]) - w/2
                y1 = float(objects[2]) - h/2
                x2 = float(objects[1]) + w/2
                y2 = float(objects[2]) + h/2

                bboxs.append([y1, x1, y2, x2])

        images.append(image)
        bboxes.append(bboxs)
        classes_ids.append(class_ids)

    images = np.array(images, dtype = np.dtype('O'))
    bboxes = np.array(bboxes, dtype = np.dtype('O'))
    classes_ids = np.array(classes_ids, dtype = np.dtype('O'))

    return images, bboxes, classes_ids

train_images, train_bboxes, train_classes_ids = make_npdata(train_files)
test_images, test_bboxes, test_classes_ids = make_npdata(test_files)

np.save("/content/drive/MyDrive/SSD/np_data/train_images.npy", train_images)
np.save("/content/drive/MyDrive/SSD/np_data/train_bboxes.npy", train_bboxes)
np.save("/content/drive/MyDrive/SSD/np_data/train_classes_ids.npy", train_classes_ids)
np.save("/content/drive/MyDrive/SSD/np_data/test_images.npy", test_images)
np.save("/content/drive/MyDrive/SSD/np_data/test_bboxes.npy", test_bboxes)
np.save("/content/drive/MyDrive/SSD/np_data/test_classes_ids.npy", test_classes_ids)

train_images = tf.convert_to_tensor(train_images)
train_bboxes = tf.ragged.constant(train_bboxes)
train_classes_ids = tf.ragged.constant(train_classes_ids)
test_images = tf.convert_to_tensor(test_images)
test_bboxes = tf.ragged.constant(test_bboxes)
test_classes_ids = tf.ragged.constant(test_classes_ids)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_bboxes, train_classes_ids))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_bboxes, test_classes_ids))

def parse_dataset(image, bboxs, category_ids):
    image = tf.io.decode_jpeg(image, channels=3)
    return image, bboxs, category_ids

parsed_dataset = test_dataset.map(parse_dataset)

for image, bboxs, category_ids in parsed_dataset.take(1):
    print(f"Image shape: {image.shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(image.numpy())
    plt.show()

    print(bboxs)
    print(category_ids)

id_to_labels = {0:"document_type",
                1:"invoice_number",
                2:"account_number",
                3:"po_number",
                4:"invoice_date",
                5:"due_date",
                6:"from_date",
                7:"to_date",
                8:"supplier_name",
                9:"supplier_address",
                10:"bill_to_name",
                11:"bill_to_address",
                12:"total_tax_amount",
                13:"invoice_amount",
                14:"currency",
                15:"line_items",
                16:"table_column",
                17:"payment_terms",
                18:"nil"}

def visualize_ground_truths(
    image, boxes, classes, figsize=(10, 15), linewidth=1, color=[0, 0, 1]
):
    """Visualize Ground Truths"""
    image = np.array(image, dtype=np.uint8)
    height, width = image.shape[0], image.shape[1]
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls in zip(boxes, classes):
        text = "{}".format(_cls)
        y1, x1, y2, x2 = box
        y1 = y1*height
        x1 = x1*width
        y2 = y2*height
        x2 = x2*width
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth 
        )
        ax.add_patch(patch)
        ax.text(
            x1-3,
            y1-3,
            text,
            color = "red",
            # bbox={"facecolor": color, "alpha": 0.4, },
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

for image, bboxs, category_ids in parsed_dataset.take(1):
    image = tf.cast(image, dtype=tf.float32)
    bboxs = bboxs
    class_names = [
        id_to_labels[int(x)] for x in category_ids
    ]
    visualize_ground_truths(
        image,
        bboxs,
        class_names
    )

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import albumentations as A

import matplotlib.pyplot as plt
from tqdm import tqdm

import gc

import cv2

train_transform = A.Compose([
        A.InvertImg(p=1),
        A.LongestMaxSize(max_size=640, p=1),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode= cv2.BORDER_CONSTANT, value=0, p=1),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=0, border_mode= cv2.BORDER_CONSTANT, value=0, p=0.98),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.95)
    ],
    bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['category_ids'])
)

test_transform = A.Compose([
        A.InvertImg(p=1),
        A.LongestMaxSize(max_size=640, p=1),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode= cv2.BORDER_CONSTANT, value=0, p=1)
    ],
    bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['category_ids']),
)

def swap_xy(boxes):
    return np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

images_train = []
bboxes_train = []
classes_ids_train = []

images_test = []
bboxes_test= []
classes_ids_test = []

for t_images, t_bboxes, t_classes_ids in [(train_images, train_bboxes, train_classes_ids), (test_images, test_bboxes, test_classes_ids)]:
    for image, bboxes, classes_ids in tqdm(zip(t_images, t_bboxes, t_classes_ids), position=0, leave=True):
        image = tf.io.decode_jpeg(image).numpy()
        bboxes = swap_xy(np.array(bboxes))
        data = {"image":image, 'bboxes': bboxes, 'category_ids': classes_ids}
        for i in range(8):
            aug_data = train_transform(**data)
            images_train.append(tf.io.encode_jpeg(aug_data['image']).numpy())
            bboxes_train.append(np.array(aug_data['bboxes']))
            classes_ids_train.append(np.array(aug_data['category_ids']))
        for i in range(2):
            aug_data = train_transform(**data)
            images_test.append(tf.io.encode_jpeg(aug_data['image']).numpy())
            bboxes_test.append(np.array(aug_data['bboxes']))
            classes_ids_test.append(np.array(aug_data['category_ids']))

images_train = np.array(images_train, dtype = np.dtype('O'))
bboxes_train = np.array(bboxes_train, dtype = np.dtype('O'))
classes_ids_train = np.array(classes_ids_train, dtype = np.dtype('O'))

images_test = np.array(images_test, dtype = np.dtype('O'))
bboxes_test = np.array(bboxes_test, dtype = np.dtype('O'))
classes_ids_test = np.array(classes_ids_test, dtype = np.dtype('O'))

np.save("train.npy", np.array([images_train, bboxes_train, classes_ids_train], dtype = np.dtype('O')))
np.save("test.npy", np.array([images_test, bboxes_test, classes_ids_test], dtype = np.dtype('O')))


import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from tqdm import tqdm

import gc

import cv2

train = np.load("train.npy", allow_pickle=True)
test= np.load("test.npy", allow_pickle=True)

train_images = train[0]
train_bboxes = train[1]
train_classes_ids = train[2]

train_images = tf.convert_to_tensor(train_images)
train_bboxes = tf.ragged.constant(train_bboxes)
train_classes_ids = tf.ragged.constant(train_classes_ids)

test_images = test[0]
test_bboxes = test[1]
test_classes_ids = test[2]

test_images = tf.convert_to_tensor(test_images)
test_bboxes = tf.ragged.constant(test_bboxes)
test_classes_ids = tf.ragged.constant(test_classes_ids)

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def parse_dataset(image, bboxs, category_ids):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    bboxs = bboxs.to_tensor()
    bboxs = tf.cast(bboxs, dtype=tf.float32)
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    category_ids = tf.cast(category_ids, dtype=tf.int32)
    bboxes = tf.stack(
        [
            bboxs[:, 0] * image_shape[1],
            bboxs[:, 1] * image_shape[0],
            bboxs[:, 2] * image_shape[1],
            bboxs[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bboxes = convert_to_xywh(bboxes)
    return image, bboxs, category_ids

def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [15.0, 30.0, 60.0, 120.0, 240.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.2, ignore_iou=0.1
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()

num_classes = 19
batch_size = 8 * 8
val_batch_size = 4 * 8
autotune = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_bboxes, train_classes_ids))
val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_bboxes, test_classes_ids))

label_encoder = LabelEncoder()

train_dataset = train_dataset.map(parse_dataset, num_parallel_calls=autotune).cache()
train_dataset = train_dataset.shuffle(8*batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padded_shapes=([640, 640, 3],[None, 4], [None,]), padding_values=(0.0, 1e-8, -1),  drop_remainder=True 
)
train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(parse_dataset, num_parallel_calls=autotune).cache()
val_dataset = val_dataset.padded_batch(
    batch_size=val_batch_size, padded_shapes=([640, 640, 3],[None, 4], [None,]), padding_values=(0.0, 1e-8, -1),  drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.prefetch(autotune)

id_to_labels = {0:"document_type",
                1:"invoice_number",
                2:"account_number",
                3:"po_number",
                4:"invoice_date",
                5:"due_date",
                6:"from_date",
                7:"to_date",
                8:"supplier_name",
                9:"supplier_address",
                10:"bill_to_name",
                11:"bill_to_address",
                12:"total_tax_amount",
                13:"invoice_amount",
                14:"currency",
                15:"line_items",
                16:"table_column",
                17:"payment_terms",
                18:"nil"}

def visualize_ground_truths(
    image, boxes, classes, figsize=(10, 15), linewidth=1, color=[0, 0, 1]
):
    """Visualize Ground Truths"""
    image = np.array(image, dtype=np.uint8)
    height, width = image.shape[0], image.shape[1]
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls in zip(boxes, classes):
        text = "{}".format(_cls)
        x1, y1, x2, y2 = box
        y1 = y1*height
        x1 = x1*width
        y2 = y2*height
        x2 = x2*width
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth 
        )
        ax.add_patch(patch)
        ax.text(
            x1-3,
            y1-3,
            text,
            color = "red",
            # bbox={"facecolor": color, "alpha": 0.4, },
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_bboxes, test_classes_ids))
test_dataset = test_dataset.map(parse_dataset, num_parallel_calls=autotune)

for image, bboxs, category_ids in test_dataset.take(1):
    image = tf.cast(image, dtype=tf.float32)
    bboxs = bboxs
    class_names = [
        id_to_labels[int(x)] for x in category_ids
    ]
    visualize_ground_truths(
        image,
        bboxs,
        class_names
    )

label_encoder = LabelEncoder()
test_dataset = test_dataset.padded_batch(
    batch_size=val_batch_size, padded_shapes=([640, 640, 3],[None, 4], [None,]), padding_values=(0.0, 1e-8, -1),  drop_remainder=True
)
test_dataset = test_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
test_dataset = test_dataset.prefetch(autotune)

def return_labels(batch_images, batch_labels):
    return batch_labels

def id_label(labels):
    return int(labels[4] != -1)

test_labels = test_dataset.map(return_labels)
test_labels = test_labels.unbatch().unbatch().map(id_label)


from sklearn.cluster import KMeans
from math import sqrt

aspect_ratios = []
areas = []

for t_bboxes in [train_bboxes, test_bboxes]:
    for sample in t_bboxes:
        for box in sample:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            aspect_ratios.append(w/h)
            areas.append(sqrt(w*h*640*640))

aspect_ratios = np.array(aspect_ratios)#.reshape(-1,1)
areas = np.array(areas)#.reshape(-1,1)

areas_kmeans = KMeans(n_clusters=5, n_init=100, max_iter=500).fit(areas)
ratios_kmeans = KMeans(n_clusters=3, n_init=1000, max_iter=5000).fit(aspect_ratios)