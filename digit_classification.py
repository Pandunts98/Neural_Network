"""
We are using well known *MNIST*(http://yann.lecun.com/exdb/mnist/) as our dataset.
Lets start with *cool visualization*(http://scs.ryerson.ca/~aharley/vis/). 
The most beautiful demo is the second one, if you are not familiar with 
convolutions you can return to it in further lectures lectures. 
"""

import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')
from modules import Dense, Sequential, Sigmoid, ReLU, SoftMax, Dropout, BatchMeanSubtraction, Tanh, SoftPlus
from criterions import MSECriterion, MultiLabelCriterion, CrossEntropyCriterion
from optimizers import sgd_momentum, adam_optimizer

from utils.data_generator import generate_cat_eye, generate_spirale, generate_two_classes
from utils.batch_generator import get_batches
from utils.metrics import accuracy_score, one_hot, export_params

import os
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetch MNIST dataset and create a local copy.
if os.path.exists('mnist.npz'):
    # data = np.load('mnist.npz',)
    with np.load('mnist.npz', 'r', allow_pickle=True) as data:
        X = data['X']
        y = data['y']
else:
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # X, y = mnist.data / 255.0, mnist.target
    np.savez('mnist.npz', X=X, y=y)

print("data shape:", X.shape, y.shape)


X, Y = X / 255, one_hot(y)
train_x, test_x, train_y, test_y = X[:60000], X[60000:], Y[:60000], Y[60000:]

#### build model
net = Sequential()
net.add(Dense(784, 400))
net.add(ReLU())
# net.add(Sigmoid())
# net.add(SoftPlus())
# net.add(Dropout())
net.add(Dense(400, 128))
net.add(ReLU())
# net.add(BatchMeanSubtraction())
# net.add(ReLU())
# net.add(Dropout())
net.add(Dense(128, 10))
net.add(SoftMax())

# criterion = MultiLabelCriterion()  # loss function
criterion = CrossEntropyCriterion()
###############################
#### optimizer config
# Iptimizer params
# optimizer_config = {'learning_rate': 1e-2, 'momentum': 0.9}
#

optimizer_config = {'alpha': 0.01, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8}
optimizer_state = {}

# Looping params
n_epoch = 10
batch_size = 1024

##############################
#### Training loop

loss_history = []
acc_history = []

for i in range(n_epoch):
    # print(f"EPOCH: {i}")
    for x_batch, y_batch in get_batches(train_x, train_y, batch_size):
        net.zeroGradParameters()

        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)

        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)

        # Update weights
        adam_optimizer(net.getParameters(),
                       net.getGradParameters(),
                       optimizer_config,
                       optimizer_state)

        loss_history.append(loss)

        acc = 100 * accuracy_score(predictions.argmax(axis=-1), y_batch.argmax(axis=-1))
        acc_history.append(acc)
    print(f"EPOCH: {i}    loss: {loss}    accuracy: {acc}")

################################
#### Visualize training statistics
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("Training loss")
plt.xlabel("#iteration")
plt.ylabel("loss")
plt.plot(loss_history, 'b')

plt.subplot(122)
plt.title("Training accuracy")
plt.xlabel("#iteration")
plt.ylabel("acc")
plt.plot(acc_history, 'r')
plt.savefig('validatin_acc.png')
plt.show()


print('Current loss: %f' % loss)

net.evaluate()
param = net.getParameters()

# dict = {}
# for i, p in enumerate(param, 0):
#     cur_dict = {}
#     for ind, cur_p in enumerate(p, 0):
#         cur_dict[ind] = cur_p.tolist()
#     dict[i] = cur_dict
# export_params(dict)

pred = net.forward(test_x)
acc = 100 * accuracy_score(pred.argmax(axis=-1), test_y.argmax(axis=-1))
print(f'test accuracy:  {acc}')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
YOUR TASKS:
- **Compare** `ReLU`,`Sigmoid`, `SoftPlus` activation functions. 
You would better pick the best optimizer params for each of them, but it is 
overkill for now. Use an architecture of your choice for the comparison and let
it be fixed.

- **Try** inserting `BatchMeanSubtraction` between `Dense` module and 
  activation functions.

- Plot the losses both from activation functions comparison and 
  `BatchMeanSubtraction` comparison on one plot. Please find a scale (log?) 
  when the lines are distinguishable, do not forget about naming the axes, 
  the plot should be goodlooking. You can submit pictures of this plots.

- Write your personal opinion on the activation functions, think about 
  computation times too. Does `BatchMeanSubtraction` help?

- **Finally**, use all your knowledge to build a super cool model on this 
  dataset, do not forget to split dataset into train and validation. Use 
  **dropout** to prevent overfitting, play with **learning rate decay**. 
  You can use **data augmentation** such as rotations, translations to boost 
  your score. Use your knowledge and imagination to train a model. 

- Print your accuracy at the end of the code. Also write down the best accuracy 
  that you could get on test setIt should be around 90%.

- Hint: logloss for MNIST should be around 0.5.

- Suggestions: it can be easyer to use jupyter notebook for experimenting,
  but final results MUST be in this file (or multiple files)

Write down all your answers at the end of the file as a comment.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Your code goes here.

# ...

# Your answers here
