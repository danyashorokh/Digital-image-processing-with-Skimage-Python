from skimage import data, io
from skimage.measure import label, regionprops
import os
import numpy as np


# activation functions init
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# class neural network
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):

        # choose activation function
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # weights init
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]
                                ))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
                            1]))-1)*0.25)

    # study
    def fit(self, X, y, learning_rate=0.2, epochs = 20000):

        # input data to matrix format
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # through hidden layers
            for l in range(len(self.weights)):
                hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                a.append(hidden_inputs)

            # error computation
            error = y[i] - a[-1][:-1]
            deltas = [error * self.activation_deriv(a[-1][:-1])]
            l = len(a) - 2

            # The last layer before the output is handled separately because of
            # the lack of bias node in output
            deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            for l in range(len(a) - 3, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1][:-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights) - 1):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta[:, :-1])

            # Handle last layer separately because it doesn't have a bias unit
            i += 1
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    # predict function
    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights)):
            temp = np.ones(a.shape[0] + 1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))
        return a



# average gray
def avg_gray(img, win_size, name):
    # input image rows and cols calculating
    rows = img.shape[0]
    cols = img.shape[1]

    res = []

    i = 1
    # for each pixel in image with step of size window
    for row in range(0, rows, win_size):
        for col in range(0, cols, win_size):
            r = 0
            g = 0
            b = 0

            # for each pixel in window
            for px in range(0, win_size):
                for py in range(0, win_size):
                    r += img[row + py, col + px ,0]
                    g += img[row + py, col + px ,1]
                    b += img[row + py, col + px ,2]

            # compute average (r,g,b) channels
            avg_r = r / (win_size ** 2)
            avg_g = g / (win_size ** 2)
            avg_b = b / (win_size ** 2)

            avg = (avg_r + avg_g + avg_b) / 3

            i += 1

            # tint each region with avg hue
            for px in range(0, win_size):
                for py in range(0, win_size):
                    # img[row + py, col + px, :] = (avg_r, avg_g, avg_b)
                    img[row + py, col + px, :] = (avg, avg, avg)

            res.append(avg)

    # io.imshow(img)
    # io.show()
    io.imsave(os.getcwd() + '/output/avg_'+ name +'.jpg', img)

    return res

# region segmentation size
win_size = 25

# for input data
X = []

# open the first pack of images
images1 = io.imread_collection(os.getcwd()+'/pool1/g*.jpg')
print("first pack size: " + str(len(images1)))

i = 1
for im in images1:
    avg = avg_gray(im, win_size, "g"+str(i))
    X.append(avg)
    i += 1

# open the first pack of images
images2 = io.imread_collection(os.getcwd()+'/pool2/f*.jpg')
print("second pack size: " + str(len(images2)))

i = 1
for im in images2:
    avg = avg_gray(im, win_size, "f"+str(i))
    X.append(avg)
    i += 1


# test image
test = io.imread(os.getcwd() + '/15.JPG')

io.imshow(test)
io.show()

test1 = avg_gray(test, win_size, "test")

# search pack
search = []
for xi in X:
    search.append(xi)
search.append(test1)

# neural network init
nn = NeuralNetwork([len(X[0]), 5 ,1], 'tanh')

# test results
y1 = [1 for i in range(len(images1))]
y2 = [0 for i in range(len(images2))]
y = y1 + y2

# study neural network with 25000 epochs
nn.fit(X, y, epochs = 25000)

# predict results
j = 1
for i in search:
    pr = round(nn.predict(i)[0])
    if pr == 1:
        res = "Gal Gadot"
    elif pr == 0:
        res = "Jake McDorman"
    else: res = "Unknown"
    print(j,nn.predict(i), res)
    j += 1

