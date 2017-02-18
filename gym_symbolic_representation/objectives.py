def minimize_y_pred(y_true, y_pred):
    return y_pred


def maximize_y_pred(y_true, y_pred):
    return -y_pred


def maximize_sign(y_true, y_pred):
    return - y_true * y_pred
