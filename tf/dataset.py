import pandas as pd
import os
import tensorflow as tf


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
    val_size = int(len(dataset) * 0.2)
    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)
    print(f'train/val: {len(train_ds)}/{len(val_ds)}')