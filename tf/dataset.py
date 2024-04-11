import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image


def show(image):
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def train_val_split(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)
    return train_ds, val_ds


if __name__ == '__main__':
    artifact_dir = 'data/flickr8k'
    annotations = pd.read_csv(os.path.join(artifact_dir, 'captions.txt'))
    image_files = [
        os.path.join(artifact_dir, "Images", image_file)
        for image_file in annotations["image"].to_list()
    ]
    for image_file in image_files:
        assert os.path.isfile(image_file)
    captions = annotations["caption"].to_list()
    print(f'len(image_files): {len(image_files)}')
    print(f'len(captions): {len(captions)}')

    dataset = tf.data.Dataset.from_tensor_slices((image_files, captions))
    dataset = dataset.shuffle(len(dataset))
    train_ds, val_ds = train_val_split(dataset, val_ratio=0.2)
    print(f'train/val: {len(train_ds)}/{len(val_ds)}')

    image_dataset = tf.data.Dataset.from_tensor_slices(image_files).map(parse_image)
    caption_dataset = tf.data.Dataset.from_tensor_slices(captions)

    image_caption_dataset = tf.data.Dataset.zip((image_dataset, caption_dataset))
    train_image_caption_ds, val_image_caption_ds = train_val_split(image_caption_dataset, val_ratio=0.2)
    train_image_caption_ds = train_image_caption_ds.shuffle(len(train_image_caption_ds))
    print(f'train_image_caption_ds/val_image_caption_ds: {len(train_image_caption_ds)}/{len(val_image_caption_ds)}')

    for image, caption in train_image_caption_ds.take(1):
        show(image)