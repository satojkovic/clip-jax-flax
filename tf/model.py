import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


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


if __name__ == "__main__":
    image_encoder = ImageEncoder(input_shape=(224, 224, 3), trainable=True)
    image_encoder.build(input_shape=(None, 224, 224, 3))
    image_encoder.summary()
