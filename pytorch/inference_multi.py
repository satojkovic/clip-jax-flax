import argparse
import torch
from model import CLIPDualEncoderModel
from datasets import load_dataset
from transformers import AutoTokenizer
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from imagenet_info import imagenet_classes, imagenet_templates


def preproc_image(image, transforms, device):
    image = transforms(image=np.array(image))['image']
    image = torch.tensor(image).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)
    return image


def zeroshot_classifier(labels, templates, model, tokenizer, device, max_length=100):
    zeroshot_weights = []
    for label in tqdm(labels):
        texts = [template.format(label) for template in templates]
        texts = tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt').to(device)
        text_embeddings = model.encode_text(texts['input_ids'], texts['attention_mask'])
        text_embeddings = text_embeddings.mean(dim=0)
        text_embeddings /= text_embeddings.norm()
        text_embeddings = text_embeddings.cpu().detach().numpy()
        zeroshot_weights.append(text_embeddings)
    zeroshot_weights = np.stack(zeroshot_weights, axis=1)
    return zeroshot_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True, help='Path to ckpt.')
    args = parser.parse_args()

    #
    # Zero-shot classification
    #

    imagenette = load_dataset(
        'frgfm/imagenette',
        '320px',
        split='validation',
        revision='4d512db'
    )

    image_encoder_alias = "resnet50"
    text_encoder_alias = "distilbert-base-uncased"

    device = 'mps'
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)
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

    # Prompt ensembling
    labels = imagenette.info.features['label'].names
    zeroshot_weights = zeroshot_classifier(labels, imagenet_templates, model, tokenizer, device)

    preds = {}
    for i in tqdm(range(len(imagenette))):
        if imagenette[i]['image'].mode != 'RGB':
            continue
        image = preproc_image(imagenette[i]['image'], transforms=transforms, device=device)
        image_embeddings = model.encode_image(image)
        image_embeddings = image_embeddings.cpu().detach().numpy()
        score = np.dot(image_embeddings, zeroshot_weights)
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