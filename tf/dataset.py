import pandas as pd
import os


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

