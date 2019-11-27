# Multilayer Perceptron project from school 42
This project aims to introduce you to artificial neural networks, and implementation of algorithms at the heart of the learning process.

	➜ pip3 install -r requirements.txt

Example of usage:

For Training:

	➜ python3 mlp_train.py -opt rmsprop adam normal nesterov adagrad -lr 0.005  -b 13 -nb -m 0.5 -bm

Then for Predictions:

	➜ python3 mlp_predict.py


Usage:

	➜ python3 mlp_train.py -h
	usage: mlp_train.py [-h] [-L LAYERS] [-U UNITS] [-lr LEARNING_RATE]
                    [-b BATCH_SIZE] [-e EPOCHS] [-la LMBD] [-o OUTLIERS]
                    [-opt {normal,adam,adagrad,nesterov,rmsprop} [{normal,adam,adagrad,nesterov,rmsprop} ...]]
                    [-s] [-m MOMENTUM] [-es] [-p PATIENCE] [-nb] [-bm]
                    [dataset]

```
positional arguments:
  dataset               a data set

optional arguments:
  -h, --help            show this help message and exit
  -L LAYERS, --layers LAYERS
                        Number of layers
  -U UNITS, --units UNITS
                        Number of units per layer
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning Rate's value
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of batch
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -la LMBD, --lmbd LMBD
                        Lambda's value for regularization
  -o OUTLIERS, --outliers OUTLIERS
                        Drop outliers with the z score given
  -opt {normal,adam,adagrad,nesterov,rmsprop} [{normal,adam,adagrad,nesterov,rmsprop} ...], --optimizations {normal,adam,adagrad,nesterov,rmsprop} [{normal,adam,adagrad,nesterov,rmsprop} ...]
                        Optimization list for Gradient descent
  -s, --shuffle         Shuffle the data set
  -m MOMENTUM, --momentum MOMENTUM
                        Momentum 's value fot NAG (Nesterov's Accelerated
                        Momentum)
  -es, --early_stopping
                        Early Stopping Activation
  -p PATIENCE, --patience PATIENCE
                        Number of epochs waited to execute early stopping
  -nb, --no_batch_too   Perform Gradient Descent also without batches (when
                        batch_size is enabled)
  -bm, --bonus_metrics  Precision, Recall and F Score metrics```
