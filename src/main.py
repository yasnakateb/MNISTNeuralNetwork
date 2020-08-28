######################################################
# Using sklearn ***only**** to get mnist dataset and 
# testing accuracy.
######################################################
from mnist_data import *
from mnist_result import *
from simple_neural_neteork import *

lr = 4
eps = .9
batch_size = 128
scale = 255
nrows = 10
train_length = 60000

mnist_data = MnistData(train_length, nrows)
snn = SimpleNeuralNetwork(lr, eps, batch_size, scale)

# 784 inputs (= 28 x 28)
data, target = mnist_data.load_data()
data = snn.normalize(data)

X_train, X_test, Y_train, Y_test = mnist_data.get_get_train_test_data(data, target)

batches = -(-train_length // batch_size)
epochs = 9

train_result = snn.train(X_train, X_test, Y_train, Y_test, epochs, batches, nrows)

parameters = train_result[0]
train_cost_list = train_result[1]
test_cost_list = train_result[2]

temp_values = snn.forward(X_test, parameters)
predictions = np.argmax(temp_values["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

mnist_result = MnistResult()

mnist_result.show_accuracy(predictions, labels)
mnist_result.show_cost(train_cost_list, test_cost_list)