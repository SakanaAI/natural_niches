from sklearn import datasets


def load_data(train_test_split=0.8):
    images, labels = datasets.load_digits(return_X_y=True)
    x_train, y_train = (
        images[: int(len(images) * train_test_split)],
        labels[: int(len(images) * train_test_split)],
    )
    x_test, y_test = (
        images[int(len(images) * train_test_split) :],
        labels[int(len(images) * train_test_split) :],
    )
    return (x_train, y_train), (x_test, y_test)
