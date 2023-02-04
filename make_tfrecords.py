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

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, objects_count, bboxs, category_ids):
    feature = {
        "image": image_feature(image),
        "objects_count": int64_feature(objects_count),
        "bboxs": float_feature_list(bboxs),
        "category_ids": int64_feature_list(category_ids),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "objects_count": tf.io.FixedLenFeature([], tf.int64),
        "bboxs": tf.io.VarLenFeature(tf.float32),
        "category_ids": tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bboxs"] = tf.sparse.to_dense(example["bboxs"])
    example["bboxs"] = tf.reshape(example["bboxs"], [example["objects_count"], 4])
    example["category_ids"] = tf.sparse.to_dense(example["category_ids"])
    return example

def make_tfrecord(data_files, record_name):
    
    with tf.io.TFRecordWriter(record_name) as writer:
        for data_file in data_files:
            image_file, annotation_file = data_file[0], data_file[1]
            image = tf.io.decode_jpeg(tf.io.read_file(image_file))

            category_ids = []
            bboxs = []
            with open(annotation_file) as txtf:
                for line in txtf.readlines():
                    objects = line.split(" ")
                    category_ids.append(int(objects[0]))

                    h= float(objects[4].rstrip("\n"))
                    w = float(objects[3])
                    x = float(objects[1])
                    y = float(objects[2])
                    x1 = float(objects[1]) - w/2
                    y1 = float(objects[2]) - h/2
                    x2 = float(objects[1]) + w/2
                    y2 = float(objects[2]) + h/2

                    bboxs += [y1, x1, y2, x2]

            objects_count = len(category_ids)

            example = create_example(image, objects_count, bboxs, category_ids)
            writer.write(example.SerializeToString())

make_tfrecord(train_files, "train.tfrec")
make_tfrecord(test_files, "test.tfrec")

raw_dataset = tf.data.TFRecordDataset("/content/train.tfrec")
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()

train_dataset = tf.data.TFRecordDataset("/content/tf_data/train.tfrec")
train_dataset = train_dataset.cache().map(parse_tfrecord_fn)

val_dataset = tf.data.TFRecordDataset("/content/tf_data/test.tfrec")
val_dataset = val_dataset.cache().map(parse_tfrecord_fn)

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

for sample in val_dataset.take(1):
    image = tf.cast(sample["image"], dtype=tf.float32)
    bboxs = sample["bboxs"]
    class_names = [
        id_to_labels[int(x)] for x in sample["category_ids"]
    ]
    visualize_ground_truths(
        image,
        bboxs,
        class_names
    )