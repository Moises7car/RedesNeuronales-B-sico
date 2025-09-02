import mnist_loader
import network

def main():
    # Cargar datos
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Definir la red: 784 neuronas de entrada, 30 ocultas, 10 de salida
    net = network.Network([784, 30, 10])

    # Entrenar con SGD
    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=0.5, test_data=test_data)

if __name__ == "__main__":
    main()
