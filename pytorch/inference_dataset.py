import argparse
import torch
from model import CLIPDualEncoderModel
from datasets import load_dataset
from transformers import AutoTokenizer
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def preproc_image(image, transforms):
    image = transforms(image=np.array(image))['image']
    image = torch.tensor(image).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True, help='Path to ckpt.')
    args = parser.parse_args()

    # Zero-shot classification

    imagenette = load_dataset(
        'frgfm/imagenette',
        '320px',
        split='validation',
        revision='4d512db'
    )
    # Create labels
    labels = imagenette.info.features['label'].names
    clip_labels = [f'a photo of a {label}' for label in labels]
    max_length = max([len(clip_label) for clip_label in clip_labels])

    image_encoder_alias = "resnet50"
    text_encoder_alias = "distilbert-base-uncased"

    device = 'mps'
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)
    label_tokens = tokenizer(
        text=clip_labels,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt').to(device)
    print(f'label tokens: {label_tokens.keys()}')

    model = CLIPDualEncoderModel.load_from_checkpoint(
        args.ckpt_path,
        image_encoder_alias=image_encoder_alias,
        text_encoder_alias=text_encoder_alias)
    model.eval()

    target_size = 224
    transforms = A.Compose(
        [
            A.Resize(target_size, target_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

    inputs = {key: value for key, value in label_tokens.items()}
    preds = {}
    for i in tqdm(range(len(imagenette))):
        if imagenette[i]['image'].mode != 'RGB':
            continue
        image = preproc_image(imagenette[i]['image'], transforms=transforms)
        inputs['image'] = image
        image_embeddings, text_embeddings = model(inputs)
        image_embeddings = image_embeddings.cpu().detach().numpy()
        text_embeddings = text_embeddings.cpu().detach().numpy()
        score = np.dot(image_embeddings, text_embeddings.T)
        pred = np.argmax(score)
        preds[i] = pred

    true_preds = []
    for i, label in enumerate(imagenette['label']):
        if i not in preds:
            continue
        if label == preds[i]:
            true_preds.append(1)
        else:
            true_preds.append(0)
    print(f'Accuracy: {sum(true_preds) / len(true_preds)}')