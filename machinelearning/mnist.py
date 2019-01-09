from typing import Dict

import keras
import numpy


class MNIST:
    height = 28
    width = 28
    classes = 10
    train_images = 60000
    test_images = 10000

    def __init__(self) -> None:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        self.x_train = self.flatten_and_normalize(self.train_images, x_train)
        self.x_test = self.flatten_and_normalize(self.test_images, x_test)

        self.y_train = keras.utils.to_categorical(y_train, self.classes)
        self.y_test = keras.utils.to_categorical(y_test, self.classes)

    def conv_net(self) -> Dict:
        model = keras.models.Sequential()

        # FIXME: I have no idea what filters means, nor do I know if
        # kernel_size and input_shape should be height/width
        model.add(keras.layers.Conv2D(
            filters=100,
            kernel_size=(self.height, self.width),
            input_shape=(self.height, self.width, 1)
        ))

        # FIXME: seems to be needed, otherwise it will fail because
        # y_train isn't 4D...  why?
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.classes))

        # FIXME: the loss and optimizer were chosen simply because they
        # were the first ones mentioned in the docs
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=["accuracy"]
        )

        model.fit(self.x_train, self.y_train, epochs=10)

        return {
            model.metrics_names[i]: value
            for i, value in enumerate(model.evaluate(self.x_test, self.y_test))
        }

    def flatten_and_normalize(self,
                              images: int,
                              data: numpy.ndarray) -> numpy.ndarray:
        """
        Flatten the images and normalize them from 0-255 to 0-1.
        """
        return                                                          \
            data.reshape(images, self.height, self.width, 1) \
            .astype("float")                                            \
            / 255
