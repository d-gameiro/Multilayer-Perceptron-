import sys
import numpy as np
import pandas as pd
import inspect
import re
import copy
import logging


def describe(arg):
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = ''.join([line.strip() for line in context])
        m = re.search(r'describe\s*\((.+?)\)$', caller_lines)
        if m:
            caller_lines = m.group(1)
            position = str(callerframeinfo.filename) + "@" + str(callerframeinfo.lineno)

            # Add additional info such as array shape or string length
            additional = ''
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "describe(" + caller_lines + ") = ", end='')
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("Describe: couldn't find caller context")

    finally:
        del frame
        del callerframeinfo


def get_data(args):
    dataset_f = "resources/dataset_test.csv"
    weights_f = "weights.npy"
    if len(args) < 3:
        print(
                "2 arguments needed. "
                "Try with \"resources/dataset_test.csv\" and \"weights.npy\".")
    else:
        dataset_f = args[1]
        weights_f = args[2]
    try:
        dataset = pd.read_csv(dataset_f, header=None)
        dataset = dataset.drop(columns=[0])
    except Exception as e:
        print("Can't extract data from {}.".format(dataset_f))
        print(e.__doc__)
        sys.exit(0)
    try:
        data_train = np.load(weights_f, allow_pickle=True)
    except Exception as e:
        print("Can't extract data from {}.".format(weights_f))
        print(e.__doc__)
        sys.exit(0)
    return dataset, data_train


def feature_scaling(df, stats):
    for subj in stats:
        df[subj] = (df[subj] - stats[subj]['mean']) / stats[subj]['std']
    return df


def layers_init(hidden_layers, units, n_features, n_class):
    i = 0
    layers = [layer(n_features)]
    while i < hidden_layers:
        layers.append(layer(units))
        i += 1
    layers.append(layer(n_class, activation='softmax')) #option
    return layers


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z), axis=0)[:, None])


def binary_cross_entropy(predict, y_class):
    size = np.size(predict, 0)
    predict = predict.reshape(-1, 2)
    return ((1 / size)
            * (-1 * y_class[:, 0].dot(np.log(predict[:, 0]))
            - (1 - y_class[:, 0]).dot(np.log(1 - predict[:, 0]))))


class layer:
    seed_id = 0
    activ_dict = {
            'sigmoid': sigmoid,
            'softmax': softmax,
    }

    def __init__(self, size, activation='sigmoid'):
        self.size = size
        self.activation = layer.activ_dict[activation]
        self.seed = layer.seed_id
        layer.seed_id += 1


def theta_init(layer_1, layer_2, seed=0, eps=0.5):
    np.random.seed(seed)
    return np.random.rand(layer_2, layer_1 + 1) * 2 * eps - eps


class network:
    def __init__(
            self, layers, data_train, data_valid=None, args=None, thetas=None):
        self.layers = layers
        self.size = len(layers)
        self.train_size = len(data_train)
        self.x = data_train.drop(columns=['class', 'vec_class']).to_numpy()
        self.batched_x = 0
        self.batched_vec_y = 0
        self.y = data_train['class'].to_numpy()
        self.vec_y = np.asarray(data_train['vec_class'].tolist())
        self.n_class = len(np.unique(self.y))
        self.early_stop_counter = 0
        self.early_stop_index = 0
        self.early_stop_min = None
        self.velocity = []
        self.thetas = []
        self.best_thetas = []
        self.deltas = []
        self.predict = []
        self.valid_predict = []
        self.best_predict = []
        unique, counts = np.unique(self.y, return_counts=True)
        self.count_y = dict(zip(unique, counts))
        if data_valid is not None:
            self.valid_size = len(data_valid)
            self.valid_x = data_valid.drop(
                    columns=['class', 'vec_class']).to_numpy()
            self.valid_y = data_valid['class'].to_numpy()
            self.valid_vec_y = np.asarray(data_valid['vec_class'].tolist())
            unique, counts = np.unique(self.valid_y, return_counts=True)
            self.count_valid_y = dict(zip(unique, counts))
            i = 0
            while i < self.size - 1:
                self.thetas.append(theta_init(
                    self.layers[i].size,
                    self.layers[i + 1].size,
                    self.layers[i].seed))
                self.best_thetas.append(theta_init(
                    self.layers[i].size, self.layers[i + 1].size, eps=0.0))
                self.deltas.append(theta_init(
                    self.layers[i].size, self.layers[i + 1].size, eps=0.0))
                self.velocity.append(theta_init(
                    self.layers[i].size, self.layers[i + 1].size, eps=0.0))
                i += 1
        else:
            self.thetas = copy.deepcopy(thetas)
        if args is not None:
            self.lmbd = args.lmbd
            self.momentum = args.momentum
            self.patience = args.patience

    def split(self, batch_size):
        sections = []
        index = batch_size
        while index + batch_size <= self.train_size:
            sections.append(index)
            index += batch_size
        self.batched_x = np.split(self.x, sections)
        self.batched_vec_y = np.split(self.vec_y, sections)

    def early_stopping(self, val_costs, index):
        if self.early_stop_min is None:
            self.early_stop_min = val_costs[index]
        if self.early_stop_min >= val_costs[index]:
            self.early_stop_min = val_costs[index]
            self.early_stop_counter = 0
            self.best_thetas = copy.deepcopy(self.thetas)
            self.best_predict = copy.deepcopy(self.valid_predict)
            self.early_stop_index = index
            return 0
        elif self.patience > self.early_stop_counter:
            self.early_stop_counter += 1
            return 0
        else:
            return 1


def forward_pro(net, row):
    i = 0
    a = [row.reshape(-1, 1)]
    b = np.array([[1.0]]).reshape(1, 1)
    while i < net.size - 1:
        a[i] = np.concatenate((b, a[i]), axis=0)
        a.append(net.layers[i+1].activation(
            net.thetas[i].dot(a[i])))
        i += 1
    net.predict.append(a[i])


def prediction(net):
    i = 0
    while i < net.train_size:
        forward_pro(net, net.x[i])
        i += 1


def display_results(p, y, vec_y):
    y_predict = p.argmax(axis=1)
    i = 0
    good = 0
    size = len(y)
    ok = "\x1b[1;32;40m"
    no = "\x1b[1;31;40m"
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    pos = 0
    neg = 0
    while i < size:
        if y[i] == y_predict[i]:
            if y[i] == 1:
                true_positive += 1
                pos += 1
            else:
                true_negative += 1
                neg += 1
            good += 1
            logging.info(ok + "({},{}) - row[{} {}]".format(
                y[i], y_predict[i], p[i, 0], p[i, 1]) + "\x1b[0m")
        else:
            if y[i] == 1:
                false_negative += 1
                pos += 1
            else:
                false_positive += 1
                neg += 1
            logging.info(no + "({},{}) - row[{} {}]".format(
                y[i], y_predict[i], p[i, 0], p[i, 1]) + "\x1b[0m")
        i += 1
    try:
        precision = float(true_positive/(true_positive + false_positive))
    except Exception as e:
        precision = 0
        print(e.__doc__)
    try:
        recall = float(true_positive/(true_positive + false_negative))
    except Exception as e:
        recall = 0
        print(e.__doc__)
    try:
        f_score = 2 * (precision * recall/(precision + recall))
    except Exception as e:
        f_score = 0
        print(e.__doc__)
    logging.info("loss (binary crossentropy) : {}".format(
        binary_cross_entropy(p, vec_y)))
    logging.info("Correctly Predicted : {}/{}".format(good, size))
    logging.info(ok + "True Positive : {}/{}".format(
        true_positive, pos) + "\x1b[0m")
    logging.info(ok + "True Negative : {}/{}".format(
        true_negative, neg) + "\x1b[0m")
    logging.info(no + "False Positive : {}/{}".format(
        false_positive, neg) + "\x1b[0m")
    logging.info(no + "False Negative : {}/{}".format(
        false_negative, pos) + "\x1b[0m")
    logging.info("Precision = {}".format(precision))
    logging.info("Recall = {}".format(recall))
    logging.info("F Score = {}\n".format(f_score))


def pre_process(df, stats):
    df = feature_scaling(df, stats)
    df = df.rename(columns={1: "class"})
    df['class'] = df['class'].map({'M': 1, 'B': 0})
    df['vec_class'] = df['class'].map({1: [0, 1], 0: [1, 0]})
    return df


def run_prediction(df, layers, weights):
    for weight in weights:
        logging.info(
                "\x1b[1;33;40mPrediction with {} gradient "
                "( Batched Size = {} ).\x1b[0m".format(
                    weight['type'], weight['batch_size']))
        net = network(layers, df, thetas=weight['thetas'])
        prediction(net)
        display_results(np.asarray(net.predict), net.y, net.vec_y)


def init_logging():
    logfile = 'predictions.log'
    try:
        level = logging.INFO
        format = '%(message)s'
        handlers = [
                logging.FileHandler(logfile),
                logging.StreamHandler()]
    except Exception as e:
        print("Can't write to {}.".format(logfile))
        print(e.__doc__)
        sys.exit(0)
    logging.basicConfig(level=level, format=format, handlers=handlers)


def main():
    init_logging()
    df, data_train = get_data(sys.argv)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', len(df))
    df = pre_process(df, data_train[0]['stats'])
    layers = layers_init(
            data_train[0]['layers'],
            data_train[0]['units'],
            len(df.columns) - 2, 2)
    run_prediction(df, layers, data_train[0]['weights'])


if __name__ == '__main__':
    main()
