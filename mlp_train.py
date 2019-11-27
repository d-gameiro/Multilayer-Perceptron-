import sys
import numpy as np
import pandas as pd
import inspect
import re
import math
import matplotlib.pyplot as plt
import copy
import timeit
import argparse
import logging
from datetime import datetime


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


# Args Check
def check_fpositive(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
                "%s is an invalid positive value" % value)
    return fvalue


def check_fpositive_null(value):
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(
                "%s is an invalid positive or null value" % value)
    return fvalue


def check_ipositive(value):
    ivalue = int(value)
    if ivalue <= 0 or ivalue > 1000000:
        raise argparse.ArgumentTypeError(
                "%s is an invalid positive value" % value)
    return ivalue


def check_ipositive_null(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 1000000:
        raise argparse.ArgumentTypeError(
                "%s is an invalid positive or null value" % value)
    return ivalue


def check_outliers(value):
    fvalue = float(value)
    if fvalue < 2.0 or fvalue > 4.0:
        raise argparse.ArgumentTypeError(
                "%s is an invalid z score (must be 2 < z <= 4)" % value)
    return fvalue


# Get Data by argguments
def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "dataset",
            help="a data set",
            nargs='?',
            default="resources/dataset_train.csv")
    parser.add_argument(
            "-L",
            "--layers",
            help="Number of layers",
            type=check_ipositive,
            default=2)
    parser.add_argument(
            "-U",
            "--units",
            help="Number of units per layer",
            type=check_ipositive,
            default=12)
    parser.add_argument(
            "-lr",
            "--learning_rate",
            help="Learning Rate's value",
            type=check_fpositive,
            default=1.0)
    parser.add_argument(
            "-b",
            "--batch_size",
            help="Size of batch",
            type=check_ipositive,
            default=0)
    parser.add_argument(
            "-e",
            "--epochs",
            help="Number of epochs",
            type=check_ipositive,
            default=80)
    parser.add_argument(
            "-la",
            "--lmbd",
            help="Lambda's value for regularization",
            type=check_fpositive_null,
            default=0.0)
    parser.add_argument(
            "-o",
            "--outliers",
            help="Drop outliers with the z score given",
            type=check_outliers,
            default=0.0)
    parser.add_argument(
            '-opt',
            '--optimizations',
            nargs='+',
            help='Optimization list for Gradient descent',
            choices=['normal', 'adam', 'adagrad', 'nesterov', 'rmsprop'],
            default=['normal'])
    parser.add_argument(
            "-s",
            "--shuffle",
            help="Shuffle the data set",
            action="store_true")
    parser.add_argument(
            "-m",
            "--momentum",
            help="Momentum 's value fot NAG (Nesterov's Accelerated Momentum)",
            type=check_fpositive,
            default=0.01)
    parser.add_argument(
            "-es",
            "--early_stopping",
            help="Early Stopping Activation",
            action="store_true")
    parser.add_argument(
            "-p",
            "--patience",
            help="Number of epochs waited to execute early stopping",
            type=check_ipositive_null,
            default=0)
    parser.add_argument(
            "-nb",
            "--no_batch_too",
            help=(
                "Perform Gradient Descent also without batches "
                "(when batch_size is enabled)"),
            action="store_true")
    parser.add_argument(
            "-bm",
            "--bonus_metrics",
            help="Precision, Recall and F Score metrics",
            action="store_true")
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.dataset, header=None)
        data = data.drop(columns=[0])
    except Exception as e:
        print("Can't extract data from {}.".format(args.dataset))
        print(e.__doc__)
        sys.exit(0)
    return data, args


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z), axis=0))


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
        self.value = 0
        layer.seed_id += 1


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


def display_results(costs, valid_costs, epochs):
    if all(costs[i] >= costs[i+1] for i in range(epochs-1)):
        logging.info(
                '\x1b[1;32;40m'
                + 'Train : Cost always decrease.'
                + '\x1b[0m')
    else:
        logging.info(
                '\x1b[1;31;40m'
                + 'Train : Cost don\'t always decrease'
                + ' (Try smaller Learning Rate ?).'
                + '\x1b[0m')
    if all(valid_costs[i] >= valid_costs[i+1] for i in range(epochs-1)):
        logging.info(
                '\x1b[1;32;40m'
                + 'Valid : Cost always decrease.'
                + '\x1b[0m')
    else:
        logging.info(
                '\x1b[1;31;40m'
                + 'Valid : Cost don\'t always decrease.'
                + '\x1b[0m')
    logging.info("train cost = {}".format(costs[epochs]))
    logging.info("valid cost = {}".format(valid_costs[epochs]))


def prediction(net):
    forward_propagation(net, train=False)
    forward_propagation(net)


class gradient_descent:

    def normal(self, net, derivate):
        t = 0
        while t < net.size-1:
            net.thetas[t] = net.thetas[t] - self.lr * derivate[t]
            t += 1
        return net.thetas

    def rmsprop(self, net, derivate):
        t = 0
        while t < net.size - 1:
            self.cache[t] = (
                    self.decay_rate
                    * self.cache[t]
                    + (1 - self.decay_rate)
                    * derivate[t]**2)
            net.thetas[t] = (
                    net.thetas[t]
                    - self.lr * derivate[t]
                    / (np.sqrt(self.cache[t]) + self.eps))
            t += 1
        return net.thetas

    def adagrad(self, net, derivate):
        t = 0
        while t < net.size - 1:
            self.cache[t] += derivate[t]**2
            net.thetas[t] = (
                    net.thetas[t]
                    - self.lr * derivate[t]
                    / (np.sqrt(self.cache[t]) + self.eps))
            t += 1
        return net.thetas

    def adam(self, net, derivate):
        t = 0
        while t < net.size - 1:
            self.m[t] = self.beta1 * self.m[t] + (1 - self.beta1) * derivate[t]
            self.v[t] = (
                    self.beta2
                    * self.v[t]
                    + (1 - self.beta2) * (derivate[t]**2))
            net.thetas[t] = (
                    net.thetas[t]
                    - self.lr
                    * self.m[t] / (np.sqrt(self.v[t]) + self.eps))
            t += 1
        return net.thetas

    def nesterov(self, net, derivate):
        t = 0
        cache = copy.deepcopy(net.velocity)
        while t < net.size - 1:
            net.velocity[t] = (
                    net.momentum * net.velocity[t]
                    - self.lr * derivate[t])
            net.thetas[t] = (
                    net.thetas[t]
                    - net.momentum * cache[t]
                    + ((1 + net.momentum) * net.velocity[t]))
            t += 1
        return net.thetas

    optimizations = {
            'normal': normal,
            'nesterov': nesterov,
            'adagrad': adagrad,
            'adam': adam,
            'rmsprop': rmsprop,
    }

    def __init__(self, net, args, optimization='normal', batched=0):
        self.net = net
        self.args = args
        self.batch_size = batched
        self.epochs = args.epochs
        self.lr = args.learning_rate
        self.optimization = gradient_descent.optimizations[optimization]
        self.n_batch = None
        self.opt = optimization
        self.valid_metrics = {'correct': []}
        self.valid_metrics['precision'] = []
        self.valid_metrics['recall'] = []
        self.valid_metrics['f_score'] = []
        self.metrics = {'correct': []}
        self.metrics['precision'] = []
        self.metrics['recall'] = []
        self.metrics['f_score'] = []
        if optimization == 'rmsprop' or optimization == 'adagrad':
            self.decay_rate = 0.9
            self.eps = 0.00001
            self.cache = copy.deepcopy(net.deltas)
        elif optimization == 'adam':
            self.eps = 0.00000001
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.m = copy.deepcopy(net.deltas)
            self.v = copy.deepcopy(net.deltas)

    def perform(self):
        info = (
                "\x1b[1;33;40m{} Type: {} | layers: {} | units: {}"
                " | learning rate : {} | epochs {} | batch size: {}"
                " | lambda: {} | early stop: {} | patience: {}")
        rms = " | epsilon {} | decay rate {}"
        adam = " | epsilon {} | beta 1 {} | beta 2 {}"
        nes = " | momentum: {}"
        end_color = "\x1b[0m"
        if self.opt == 'rmsprop' or self.opt == 'adagrad':
            logging.info((info + rms + end_color).format(
                datetime.today().strftime('%Y-%m-%d %H:%M'), self.opt,
                self.args.layers, self.args.units, self.lr, self.epochs,
                self.batch_size, self.net.lmbd, self.args.early_stopping,
                self.args.patience, self.eps, self.decay_rate))
        elif self.opt == 'adam':
            logging.info((info + adam + end_color).format(
                datetime.today().strftime('%Y-%m-%d %H:%M'), self.opt,
                self.args.layers, self.args.units, self.lr, self.epochs,
                self.batch_size, self.net.lmbd, self.args.early_stopping,
                self.args.patience, self.eps, self.beta1, self.beta2))
        elif self.opt == 'nesterov':
            logging.info((info + nes + end_color).format(
                datetime.today().strftime('%Y-%m-%d %H:%M'), self.opt,
                self.args.layers, self.args.units, self.lr, self.epochs,
                self.batch_size, self.net.lmbd, self.args.early_stopping,
                self.args.patience, self.args.momentum))
        else:
            logging.info((info + end_color).format(
                datetime.today().strftime('%Y-%m-%d %H:%M'), self.opt,
                self.args.layers, self.args.units, self.lr, self.epochs,
                self.batch_size, self.net.lmbd, self.args.early_stopping,
                self.args.patience))
        if self.batch_size:
            self.net.split(self.args.batch_size)
            self.n_batch = len(self.net.batched_x)
            if not self.args.early_stopping:
                self.stochastic_gd()
            else:
                self.stochastic_es_gd()
        else:
            if not self.args.early_stopping:
                self.normal_gd()
            else:
                self.normal_es_gd()

    def normal_gd(self):
        self.costs = []
        self.valid_costs = []
        e = 0
        start = timeit.default_timer()
        while e < self.epochs:
            derivate = backward_propagation(self.net)
            self.add_cost(e)
            self.net.thetas = self.optimization(self, self.net, derivate)
            e += 1
        prediction(self.net)
        self.add_cost(e)
        self.net.best_thetas = copy.deepcopy(self.net.thetas)
        stop = timeit.default_timer()
        logging.info('Time Gradient: {}'.format(stop - start))
        display_fscore(np.asarray(self.net.valid_predict), self.net.valid_y)
        display_results(self.costs, self.valid_costs, self.epochs)

    def stochastic_gd(self):
        self.costs = []
        self.valid_costs = []
        e = 0
        start = timeit.default_timer()
        prediction(self.net)
        self.add_cost(e)
        while e < self.epochs:
            b = 0
            while b < self.n_batch:
                derivate = backward_batched_propagation(
                        self.net,
                        self.net.batched_x[b],
                        self.net.batched_vec_y[b])
                self.net.thetas = self.optimization(self, self.net, derivate)
                b += 1
            prediction(self.net)
            self.add_cost(e+1)
            e += 1
        self.net.best_thetas = copy.deepcopy(self.net.thetas)
        stop = timeit.default_timer()
        logging.info('Time Gradient: {}'.format(stop - start))
        display_fscore(np.asarray(self.net.valid_predict), self.net.valid_y)
        display_results(self.costs, self.valid_costs, self.epochs)

    def normal_es_gd(self):
        self.costs = []
        self.valid_costs = []
        early_stop = 0
        e = 0
        epochs = self.args.epochs
        start = timeit.default_timer()
        while e < epochs:
            derivate = backward_propagation(self.net)
            self.add_cost(e)
            if self.net.early_stopping(self.valid_costs, e):
                early_stop = 1
                break
            self.net.thetas = self.optimization(self, self.net, derivate)
            e += 1
        if not early_stop:
            prediction(self.net)
            self.add_cost(e)
            self.net.early_stopping(self.valid_costs, e)
        stop = timeit.default_timer()
        logging.info('Time Gradient: {}'.format(stop - start))
        self.epochs = self.net.early_stop_index
        display_fscore(np.asarray(self.net.best_predict), self.net.valid_y)
        display_results(self.costs, self.valid_costs, self.epochs)

    def stochastic_es_gd(self):
        self.costs = []
        self.valid_costs = []
        e = 0
        epochs = self.args.epochs
        start = timeit.default_timer()
        prediction(self.net)
        self.add_cost(e)
        while e < epochs:
            b = 0
            while b < self.n_batch:
                derivate = backward_batched_propagation(
                        self.net,
                        self.net.batched_x[b],
                        self.net.batched_vec_y[b])
                self.net.thetas = self.optimization(self, self.net, derivate)
                b += 1
            prediction(self.net)
            self.add_cost(e+1)
            if self.net.early_stopping(self.valid_costs, e+1):
                break
            e += 1
        stop = timeit.default_timer()
        logging.info('Time Gradient: {}'.format(stop - start))
        self.epochs = self.net.early_stop_index
        display_fscore(np.asarray(self.net.best_predict), self.net.valid_y)
        display_results(self.costs, self.valid_costs, self.epochs)

    def add_cost(self, e):
        new_cost = binary_cross_entropy(
                np.asarray(self.net.predict),
                self.net.vec_y,
                self.net.lmbd,
                self.net)
        new_valid_cost = binary_cross_entropy(
                np.asarray(self.net.valid_predict),
                self.net.valid_vec_y,
                0,
                self.net)
        self.costs.append(new_cost)
        self.valid_costs.append(new_valid_cost)
        metrics = 'epochs {}/{} - loss: {:.4f} - val_loss: {:.4f}'
        bonus_metrics = (
                ' | correct: {:.4f} % - val_correct: {:.4f} %'
                ' | pre: {:.4f} - val_pre: {:.4f}'
                ' | recall: {:.4f} - val_recall: {:.4f}'
                ' | fscore: {:.4f} - val_fscore: {:.4f}')
        if not self.args.bonus_metrics:
            logging.info(
                    metrics.format(
                        e, self.args.epochs, self.costs[e], self.valid_costs[e]
                        ))
        else:
            self.add_metrics(np.asarray(self.net.predict), self.net.y)
            self.add_metrics(
                    np.asarray(self.net.valid_predict),
                    self.net.valid_y,
                    valid=True)
            logging.info((metrics + bonus_metrics).format(
                e, self.args.epochs, self.costs[e], self.valid_costs[e],
                self.metrics['correct'][e], self.valid_metrics['correct'][e],
                self.metrics['precision'][e],
                self.valid_metrics['precision'][e],
                self.metrics['recall'][e], self.valid_metrics['recall'][e],
                self.metrics['f_score'][e], self.valid_metrics['f_score'][e]))

    def add_metrics(self, p, y, valid=False):
        y_predict = p.argmax(axis=0)
        i = 0
        good = 0
        if not valid:
            size = self.net.train_size
        else:
            size = self.net.valid_size
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        while i < size:
            if y[i] == y_predict[i]:
                if y[i] == 1:
                    true_positive += 1
                else:
                    true_negative += 1
                good += 1
            else:
                if y[i] == 1:
                    false_negative += 1
                else:
                    false_positive += 1
            i += 1
        try:
            precision = float(true_positive/(true_positive + false_positive))
        except Exception:
            precision = 0
        try:
            recall = float(true_positive/(true_positive + false_negative))
        except Exception:
            recall = 0
        try:
            f_score = 2 * (precision * recall/(precision + recall))
        except Exception:
            f_score = 0
        try:
            good = float(good)/float(size) * 100
        except Exception:
            good = 0
        if valid:
            self.valid_metrics['correct'].append(good)
            self.valid_metrics['precision'].append(precision)
            self.valid_metrics['recall'].append(recall)
            self.valid_metrics['f_score'].append(f_score)
        else:
            self.metrics['correct'].append(good)
            self.metrics['precision'].append(precision)
            self.metrics['recall'].append(recall)
            self.metrics['f_score'].append(f_score)

    def plot_results(self):
        title = self.opt.upper()
        i = 1
        if self.batch_size:
            title = title + " Batched ({})".format(self.batch_size)
        if self.args.bonus_metrics:
            for metric, l in self.metrics.items():
                plt.figure(i)
                plt.xlabel('No. of epochs')
                plt.ylabel(metric)
                plt.title(metric.capitalize() + " Evolution")
                p = plt.plot(
                        np.arange(self.epochs+1),
                        l[:self.epochs+1], alpha=0.5, label=title + ' Train')
                plt.plot(
                        np.arange(self.epochs+1),
                        self.valid_metrics[metric][:self.epochs+1], '--',
                        color=p[0].get_color(),
                        label=title + ' Validation')
                plt.legend()
                i += 1
        plt.figure(i)
        plt.xlabel('No. of epochs')
        plt.ylabel('Cost Function')
        plt.title("Cost Function Evolution")
        p = plt.plot(
                np.arange(self.epochs+1),
                self.costs[:self.epochs+1], alpha=0.5, label=title + ' Train')
        plt.plot(
                np.arange(self.epochs+1),
                self.valid_costs[:self.epochs+1], '--', color=p[0].get_color(),
                label=title + ' Validation')
        plt.legend()


def theta_init(layer_1, layer_2, seed=0, eps=0.5):
    np.random.seed(seed)
    return np.random.rand(layer_2, layer_1 + 1) * 2 * eps - eps


def display_fscore(p, y):
    y_predict = p.argmax(axis=0)
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
        else:
            if y[i] == 1:
                false_negative += 1
                pos += 1
            else:
                false_positive += 1
                neg += 1
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
    logging.info("F Score = {}".format(f_score))


def binary_cross_entropy(predict, y_class, lmbd, net):
    size = np.size(predict, 1)
    predict = predict.T
    regularization = 0
    if lmbd:
        i = 0
        thetas_sum = 0
        while i < net.size - 1:
            thetas_sum += np.sum(net.thetas[i] ** 2)
            i += 1
        regularization = lmbd / (2 * size) * thetas_sum
    return (((1 / size)
            * (-1 * y_class[:, 0].dot(np.log(predict[:, 0]))
            - (1 - y_class[:, 0]).dot(np.log(1 - predict[:, 0]))))
            + regularization)


def cross_entropy(predict, y_class, lmbd, net):
    Y = []
    size = np.size(predict, 0)
    predict = predict.reshape(-1, net.n_class)
    regularization = 0
    if lmbd:
        i = 0
        thetas_sum = 0
        while i < net.size - 1:
            thetas_sum += np.sum(net.thetas[i] ** 2)
            i += 1
        regularization = lmbd / (2 * size) * thetas_sum
    i = 0
    while (i < net.n_class):
        Y.append(
                -1 * y_class[:, i].T.dot(np.log(predict[:, i]))
                - (1 - y_class[:, i]).T.dot(np.log(1 - predict[:, i])))
        i += 1
    return (1 / size) * (sum(Y)) + regularization


def get_stats(df):
    stats = {
            column: escribe(sub_dict)
            for column, sub_dict in
            df.select_dtypes(include='number').to_dict().items()}
    return stats


def percentile(percent, count, values):
    x = percent * (count - 1)
    return (values[math.floor(x)]
            + (values[math.floor(x) + 1] - values[math.floor(x)]) * (x % 1))


def escribe(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'count': count}
    stats['mean'] = sum(clean_data.values()) / count
    stats['var'] = (
            1
            / (count - 1)
            * np.sum(np.power(values - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats


def display_stats(stats):
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: ' %.6f' % x)
    res = pd.DataFrame.from_dict(stats)
    res = res.reindex([
        'count', 'mean', 'std', 'min',
        '25%', '50%', '75%', 'max'])
    print(res)


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
    layers.append(layer(n_class, activation='softmax'))
    return layers


def init_logging():
    logfile = 'metrics.log'
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


def pre_process(df, stats, args):
    df = feature_scaling(df, stats)
    df = df.rename(columns={1: "class"})
    df['class'] = df['class'].map({'M': 1, 'B': 0})
    df['vec_class'] = df['class'].map({1: [0, 1], 0: [1, 0]})
    if args.shuffle:
        df = df.sample(frac=1)
    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)
    if args.outliers:
        df_tmp = dfs[0].copy()
        dfs[0] = dfs[0][(
            np.abs(df_tmp.select_dtypes(include='number')) < args.outliers
                        ).all(axis=1)]
    return df, dfs


def create_models(net, args):
    models = []
    for opt in args.optimizations:
        models.append(gradient_descent(
            copy.deepcopy(net),
            args,
            optimization=opt,
            batched=args.batch_size))
        if args.batch_size and args.no_batch_too:
            models.append(gradient_descent(
                copy.deepcopy(net),
                args,
                optimization=opt,
                batched=0))
    return models


def run_models(models):
    weights = []
    for model in models:
        model.perform()
        model.plot_results()
        weights.append({
            'type': model.opt,
            'thetas': model.net.best_thetas,
            'batch_size': model.batch_size})
    return weights


def create_results(stats, args, weights):
    results = {'stats': stats}
    results['units'] = args.units
    results['layers'] = args.layers
    results['weights'] = weights
    outfile = 'weights.npy'
    try:
        np.save(outfile, [results])
    except Exception as e:
        print("Can't write to {}.".format(outfile))
        print(e.__doc__)
        sys.exit(0)


def derivative_calc(net, error, size):
    derivative = [0] * (net.size - 1)
    i = 0
    while i < net.size - 1:
        if not net.lmbd:
            derivative[i] = error[i] / size
        else:
            derivative[i] = (error[i] + net.lmbd * net.thetas[i])
            derivative[i][:, 0] -= (
                    net.lmbd * net.thetas[i][:, 0])
            derivative[i] /= size
        i += 1
    return derivative


def forward_propagation(net, train=True):
    # Initialization
    i = 0
    layers = net.layers
    if train:
        layers[0].value = net.x.T
        size = net.train_size
    else:
        layers[0].value = net.valid_x.T
        size = net.valid_size
    bias = np.ones((1, size))

    # Processing Forward Propagation
    while i < net.size - 1:
        layers[i].value = np.concatenate((bias, layers[i].value), axis=0)
        layers[i+1].value = layers[i+1].activation(
            net.thetas[i].dot(layers[i].value))
        i += 1

    # Saving Predictions
    if train:
        net.predict = layers[net.size-1].value
    else:
        net.valid_predict = layers[net.size-1].value
    return layers


def backward_propagation(net):
    # Forward Propagation
    forward_propagation(net, train=False)
    layers = forward_propagation(net)

    # Initialization
    delta = [0] * (net.size)
    error = [0] * (net.size)
    i = net.size - 1

    # Processing Backward Propagation
    delta[i] = layers[i].value - net.vec_y.T
    i -= 1
    while i > 0:
        error[i] = delta[i + 1].dot(layers[i].value.T)
        delta[i] = net.thetas[i].T.dot(delta[i + 1]) * layers[i].value * (1 - layers[i].value)
        delta[i] = delta[i][1:, :]  # delete the bias
        i -= 1
    error[i] = delta[i + 1].dot(layers[i].value.T)
    return derivative_calc(net, error, net.train_size)


def forward_batched_propagation(net, X, size):
    # Initialization
    i = 0
    layers = net.layers
    layers[0].value = X.T
    bias = np.ones((1, size))

    # Processing Batched Forward Propagation
    while i < net.size - 1:
        layers[i].value = np.concatenate((bias, layers[i].value), axis=0)
        layers[i+1].value = layers[i+1].activation(
            net.thetas[i].dot(layers[i].value))
        i += 1
    return layers


def backward_batched_propagation(net, x, vec_y):
    # Batched Forward Propagation
    batch_size = len(x)
    layers = forward_batched_propagation(net, x, batch_size)

    # Initialization
    delta = [0] * (net.size)
    error = [0] * (net.size)
    i = net.size - 1

    # Processing Batched Backward Propagation
    delta[i] = layers[i].value - vec_y.T
    i -= 1
    while i > 0:
        error[i] = delta[i + 1].dot(layers[i].value.T)
        delta[i] = net.thetas[i].T.dot(delta[i + 1]) * layers[i].value * (1 - layers[i].value)
        delta[i] = delta[i][1:, :]  # delete the bias
        i -= 1
    error[i] = delta[i + 1].dot(layers[i].value.T)
    return derivative_calc(net, error, batch_size)


def main():
    # log initialization
    start = timeit.default_timer()
    init_logging()

    # Get data
    df, args = get_data()

    # Basic stats calculation
    stats = get_stats(df)

    # Pre Processing
    df, dfs = pre_process(df, stats, args)

    # Neural Network's initialization
    layers = layers_init(args.layers, args.units, len(df.columns) - 2, 2)
    net = network(layers, dfs[0], dfs[1], args)
    models = create_models(net, args)

    # Train
    weights = run_models(models)

    # Model Backup
    create_results(stats, args, weights)

    # Results Visualisation
    stop = timeit.default_timer()
    logging.info(
            'Time Global: {} \n\n\n  --------------------  \n\n\n'.format(
                stop - start)
                )
    plt.show()


if __name__ == '__main__':
    main()
