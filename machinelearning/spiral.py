import csv
import itertools
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # to use matplotlib without Xorg
# pylint: disable=C0413
import matplotlib.pyplot as plt  # isort:skip  # noqa:E402
import sklearn.linear_model  # isort:skip  # noqa:E402
import sklearn.neural_network  # isort:skip  # noqa:E402
# pylint: enable=C0413


class Spiral:
    colors = ["red", "blue", "green"]

    def __init__(self, csv_file: str) -> None:
        self.input_values, self.target_values = self.read_csv(csv_file)
        self.linear_values = self.linear_classifier()
        self.nn_values = self.nn_classifier()
        self.figure = 0

    def plot_spirals(self, output: str) -> None:
        """
        Plot the spirals.
        """
        plt.figure(self.figure, figsize=(10, 10))
        self.figure += 1

        def plot(idx: int, label: str, classes: List) -> None:
            plt.subplot(2, 2, idx)
            plt.xlabel(label)
            for [x, y], cls in zip(self.input_values, classes):
                plt.scatter(x, y, c=self.colors[cls])

        plot(1, "target values", self.target_values)
        plot(2, "linear classifier", self.linear_values)
        plot(3, "neural network classifier", self.nn_values)

        plt.tight_layout()
        plt.savefig(output)

    def plot_confusion_matrices(self, output: str) -> None:
        """
        Retrieve the confusion matrix for the linear and NN classifiers.
        """
        plt.figure(self.figure, figsize=(10, 10))
        self.figure += 1

        def plot(idx: int, label: str, classes: List) -> None:
            plt.subplot(2, 2, idx)
            plt.xlabel(label)

            ticks = [0, 1, 2]
            labels = ["A", "B", "C"]
            plt.xticks(ticks, labels)
            plt.yticks(ticks, labels)

            matrix = sklearn.metrics.confusion_matrix(self.target_values,
                                                      classes)
            plt.imshow(matrix)
            # add numerical values to the plot as described on:
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
            range0 = range(matrix.shape[0])
            range1 = range(matrix.shape[1])
            for i, j in itertools.product(range0, range1):
                plt.text(j, i, matrix[i, j], color="white")

        plot(1, "linear classifier", self.linear_values)
        plot(2, "neural network classifier", self.nn_values)

        plt.tight_layout()
        plt.savefig(output)

    def linear_classifier(self) -> List:
        """
        Use a linear model to fit and predict the categories.
        """
        sgd = sklearn.linear_model.SGDClassifier()
        sgd.fit(self.input_values, self.target_values)
        return sgd.predict(self.input_values)

    def nn_classifier(self) -> List:
        """
        Use a neural network classifier to fit and predict the categories.
        """
        mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(72,),
                                                   max_iter=5000)
        mlp.fit(self.input_values, self.target_values)
        return mlp.predict(self.input_values)

    @staticmethod
    def read_csv(path: str) -> Tuple:
        """
        Read a CSV file with three fields (x, y, class).

        Return the data as a tuple of two lists; one for input values and
        one for target values.
        """
        input_values = []
        target_values = []

        with open(path) as f:
            for i, [x, y, cls] in enumerate(csv.reader(f)):  # type: ignore
                # skip the header
                if i == 0:
                    continue
                input_values.append([float(x), float(y)])
                target_values.append(int(cls))

        return (input_values, target_values)
