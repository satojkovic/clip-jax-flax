from model import CLIPDualEncoderModel
from dataset import parse_image, train_val_split, show
import pandas as pd
import os
import tensorflow as tf


if __name__ == '__main__':
    config = {
        'BATCH_SIZE': 8,
        'image_encoder_alias': 'microsoft/resnet-50',
        'text_encoder_alias': 'distilbert-base-uncased',
        'artifact_dir': 'data/flickr8k'
    }

    clip_model = CLIPDualEncoderModel(
        image_encoder_alias=config['image_encoder_alias'],
        text_encoder_alias=config['text_encoder_alias']
    )

    artifact_dir = config['artifact_dir']
    annotations = pd.read_csv(os.path.join(artifact_dir, 'captions.txt'))
    image_files = [
        os.path.join(artifact_dir, "Images", image_file)
        for image_file in annotations["image"].to_list()
    ]
    for image_file in image_files:
        assert os.path.isfile(image_file)
    captions = annotations["caption"].to_list()

    image_dataset = tf.data.Dataset.from_tensor_slices(image_files).map(parse_image)
    caption_dataset = tf.data.Dataset.from_tensor_slices(captions)

    image_caption_dataset = tf.data.Dataset.zip((image_dataset, caption_dataset))
    train_image_caption_ds, val_image_caption_ds = train_val_split(image_caption_dataset, val_ratio=0.2)
    train_image_caption_ds = train_image_caption_ds.shuffle(len(train_image_caption_ds))\
        .batch(config['BATCH_SIZE'], drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)\
        .prefetch(tf.data.AUTOTUNE)
    val_image_caption_ds = val_image_caption_ds.shuffle(len(val_image_caption_ds))\
        .batch(config['BATCH_SIZE'], drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)\
        .prefetch(tf.data.AUTOTUNE)

