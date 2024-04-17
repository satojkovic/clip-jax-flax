import argparse
import torch
from model import CLIPDualEncoderModel
from datasets import load_dataset
from transformers import AutoTokenizer


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)
    label_tokens = tokenizer(
        text=clip_labels,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt').to(device)
    print(label_tokens['input_ids'][0][:10])

    model = CLIPDualEncoderModel.load_from_checkpoint(
        args.ckpt_path,
        image_encoder_alias=image_encoder_alias,
        text_encoder_alias=text_encoder_alias)
    model.eval()
