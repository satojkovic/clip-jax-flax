import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import TFResNetModel, AutoImageProcessor
from datasets import load_dataset


class ImageEncoder(tf.keras.Model):
    def __init__(self, image_encoder_alias="microsoft/renset-50"):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(image_encoder_alias)
        self.model = TFResNetModel.from_pretrained(image_encoder_alias)
        self.target_token_idx = 0

    def call(self, x):
        inputs = self.image_processor(x, return_tensors='tf')
        outputs = self.model(**inputs)
        pooler_output = outputs.pooler_output
        # [batch, feat_dim, 1, 1] => [batch, feat_dim]
        pooler_output = tf.squeeze(pooler_output, [2, 3])
        return pooler_output


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
        return last_hidden_state[:, self.target_token_idx, :]


if __name__ == "__main__":
    image_encoder = ImageEncoder(image_encoder_alias="microsoft/resnet-50")
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_last_hidden_state = image_encoder(image)
    print(f'image last_hidden_state: {image_last_hidden_state.shape}')

    text_encoder = TextEncoder(text_encoder_alias="distilbert-base-uncased")
    input_text = "This is an example text."
    text_last_hidden_state = text_encoder(input_text)
    print(f'text last_hidden_state: {text_last_hidden_state.shape}')