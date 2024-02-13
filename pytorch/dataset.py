import torch
from typing import Optional
import albumentations as A
from abc import abstractmethod
import cv2
import pandas as pd
import os


class ImageRetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        artifact_dir: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 200,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__()
        self.artifact_dir = artifact_dir
        self.target_size = target_size
        self.image_files, self.captions = self.fetch_dataset()
        self.lazy_loading = lazy_loading
        self.images = self.image_files

        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.tokenized_captions = tokenizer(
            list(self.captions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.transforms = A.Compose(
            [
                A.Resize(target_size, target_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.tokenized_captions.items()}
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item


class Flickr8kDataset(ImageRetrievalDataset):
    def __init__(
        self,
        artifact_dir: str,
        tokenizer=None,
        target_size: int | None = None,
        max_length: int = 200,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__(artifact_dir, tokenizer, target_size, max_length, lazy_loading)

    def fetch_dataset(self):
        annotations = pd.read_csv(self.artifact_dir, "captions.txt")
        image_files = [
            os.path.join(self.artifact_dir, "Images", image_file)
            for image_file in annotations["image"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations["caption"].to_list()
        return image_files, captions
