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
        self.model = TFDistilBertModel.from_pretrained(text_encoder_alias)
        self.target_token_idx = 0

    def call(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(tf.keras.Model):
    def __init__(self, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = tf.keras.layers.Dense(projection_dim, activation='gelu')
        self.fc = tf.keras.layers.Dense(projection_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        projected = self.projection(x)
        x = self.fc(projected)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class CLIPDualEncoderModel(tf.keras.Model):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        projection_dims: int = 256,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(image_encoder_alias)
        self.text_encoder = TextEncoder(text_encoder_alias)
        self.image_projection = ProjectionHead(projection_dim=projection_dims, dropout=dropout)
        self.text_projection = ProjectionHead(projection_dim=projection_dims, dropout=dropout)

    def call(self, inputs):
        image_features = self.image_encoder(inputs["image"])
        text_features = self.text_encoder(
            input_ids=input["input_ids"], attention_mask=input["attention_mask"]
        )
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings


if __name__ == "__main__":
    image_encoder = ImageEncoder(image_encoder_alias="microsoft/resnet-50")
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_last_hidden_state = image_encoder(image)
    print(f'image last_hidden_state: {image_last_hidden_state.shape}')

    text_encoder = TextEncoder(text_encoder_alias="distilbert-base-uncased")
    input_text = "This is an example text."
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(input_text, return_tensors='tf')
    text_last_hidden_state = text_encoder(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'])
    print(f'text last_hidden_state: {text_last_hidden_state.shape}')

    image_embedding_dims = 2048
    projection_dims = 256
    dropout = 0.0
    image_projection = ProjectionHead(projection_dims, dropout)
    x = image_projection(image_last_hidden_state)
    print(f'image projection: {x.shape}')