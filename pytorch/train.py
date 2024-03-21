from model import CLIPDualEncoderModel
from dataset import ImageRetrievalDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


if __name__ == "__main__":
    image_encoder_alias = "resnet50"
    text_encoder_alias = "distilbert-base-uncased"

    model = CLIPDualEncoderModel(
        image_encoder_alias=image_encoder_alias,
        text_encoder_alias=text_encoder_alias,
    )

    data_module = ImageRetrievalDataModule(
        artifact_dir="data/flickr8k",
        dataset_name="flickr8k",
        tokenizer_alias=text_encoder_alias,
        lazy_loading=True,
    )

    logger = WandbLogger(project="CLIP", log_model="all")
    model_checkpoint = ModelCheckpoint()
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=20,
        log_every_n_steps=1,
        callbacks=[model_checkpoint, lr_monitor],
    )
    trainer.fit(model, data_module)
