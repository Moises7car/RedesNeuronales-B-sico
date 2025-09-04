# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Mejora de inicialización de pesos, la varianza es proporcional al inverso de la raíz del número de neuronas de entrada, evita la saturación continua de neuronas.
        self.weights = [np.random.randn(y, x) / np.sqrt(x) 
                        for x, y in zip(sizes[:-1], sizes[1:])] 
        
        self.cost = SoftmaxCrossEntropyCost
        
        # Inicialización de momentos para Adam. Creamos para cada matriz de pesos y de bias dos matrices del mismo tamaño pero de ceros, se van a almacenar los primeros y segundos momentos que se vayan actualizando.
        
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.R_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.R_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # contador de pasos de Adam

         
    def feedforward(self, a):
        """Todas las capas ocultas usan sigmoide, la última usa softmax."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")


    def update_mini_batch(self, mini_batch, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        #Actualiza pesos y bias usanso Adam. Tomamos los valores recomendados para beta 1 y beta 2 vistos en clase.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  

        #tomamos el promedio de gradientes en el mini batch
        nabla_b = [nb / len(mini_batch) for nb in nabla_b]
        nabla_w = [nw / len(mini_batch) for nw in nabla_w]
        self.t += 1  #incrementamos cada paso 1 a 1.
        
        for i in range(len(self.weights)):
            # Momentum para pesos (m), definición vista en clase.
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * (nabla_w[i])
            # RMS para pesos (R)
            self.R_w[i] = beta2 * self.R_w[i] + (1 - beta2) * (nabla_w[i]**2)
            
            # corrección por sesgo 
            m_w_hat = self.m_w[i] / (1-beta1**self.t)
            r_w_hat = self.R_w[i] / (1-beta2**self.t)
            
            # Actualización de pesos con las variables definidas (fórmula vista en clase)
            self.weights[i] -= eta*m_w_hat / (np.sqrt(r_w_hat) + epsilon)

            # Repetimos el procedimiento pero para los bías, es decir, combinamos momentum y RMSprop para definir la actualización de bias con las mismas fórmulas.

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * nabla_b[i]
            self.R_b[i] = beta2 * self.R_b[i] + (1 - beta2) * (nabla_b[i]**2)

            m_b_hat = self.m_b[i] / (1 - beta1**self.t)
            r_b_hat = self.R_b[i] / (1 - beta2**self.t)

            # Paso de actualización de bias
            self.biases[i] -= eta * m_b_hat / (np.sqrt(r_b_hat) + epsilon)


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        z= np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)  # usa el delta de Softmax   

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

### Función de costo Cross-Entropy (Softmax), definida como en la clase
def softmax(z):
    e_z = np.exp(z - np.max(z))   # resta max(z) para evitar overflow
    return e_z / np.sum(e_z)

class SoftmaxCrossEntropyCost(object):
    def fn(a, y):
        """Usamos la activación de la capa Softmax con una función de costo Cross Entropy."""
        return -np.sum(y * np.log(a + 1e-9))  # 1e-9 evita log(0)

    def delta(z,a,y):
        """El gradiente simplemente se obtiene con a-y, no depende de la función sigmoide primada y por lo tanto, se evitan gradientes pequeños (se aprende más rápido"""
        return (a-y) 
