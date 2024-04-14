import argparse
import torch
from model import CLIPDualEncoderModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True, help='Path to ckpt.')
    args = parser.parse_args()

    image_encoder_alias = "resnet50"
    text_encoder_alias = "distilbert-base-uncased"

    model = CLIPDualEncoderModel.load_from_checkpoint(
        args.ckpt_path,
        image_encoder_alias=image_encoder_alias,
        text_encoder_alias=text_encoder_alias)
    model.eval()