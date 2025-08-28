# Red Neuronal Básica en Python 

Este repositorio implementa una red neuronal **feedforward** con **backpropagation** y **Stochastic Gradient Descent (SGD)**.  
La red se entrena con el dataset **MNIST** 

##  Archivos principales
- `network.py` → Implementación de la red neuronal, define la clase Network la cual implementa la inicialización de pesos, la propagación hacia adelante el entrenamiento SGD (gradiente descenciente) el algoritmo backpropagation para las derivadas. 
- `mnist_loader.py` → Carga y preprocesa el dataset MNIST, devuelve los datos en un formato listo para usar con la red, en este caso un vector de 784 valores, one hot encouding.
- `mnist.pkl.gz` → Dataset comprimido de MNIST. La materia prima que necesita la red neuronal para aprender.
- `train.py` → Script de entrenamiento. Utiliza los archivos nombrados, carga los datos y ocupa network.py para definir las 784 neuronas de entrada, las 30 neuronas en la capa oculta y las 10 neuronas de salida. Se establecieron puntos importantes como el uso de 10 épocas, el tamaño de los mini-batches (10) y el learning rate que se ha tomado un poco alto en este caso (3.0). A continuación pongo el código de train.py y los resultados obtenidos, siendo una predicción muy decente.

import mnist_loader
import network

def main():
    # Cargar datos
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Definir la red: 784 neuronas de entrada, 30 ocultas, 10 de salida
    net = network.Network([784, 30, 10])

    # Entrenar con SGD
    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)

if __name__ == "__main__":
    main()



---

