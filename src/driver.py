import mnist_loader, network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, test_data=test_data)
