import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from transformers import DistilBertTokenizer, TFDistilBertModel


class ImageEncoder(tf.keras.Model):
    def __init__(self, input_shape, trainable=True):
        super().__init__()
        self.model = ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        self.model.trainable = trainable
        self.target_token_idx = 0

    def call(self, x):
        return self.model(x)


class TextEncoder(tf.keras.Model):
    def __init__(self, text_encoder_alias="distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_encoder_alias)
        self.model = TFDistilBertModel.from_pretrained(text_encoder_alias)
        self.target_token_idx = 0

    def call(self, text):
        encoded_input = self.tokenizer(text, return_tensors="tf")
        output = self.model(encoded_input)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state


if __name__ == "__main__":
    image_encoder = ImageEncoder(input_shape=(224, 224, 3), trainable=True)
    image_encoder.build(input_shape=(None, 224, 224, 3))
    image_encoder.summary()

    text_encoder = TextEncoder(text_encoder_alias="distilbert-base-uncased")
