import sys
import json
import matplotlib.pyplot as plt
import mnist_loader
import network


def test_no_regu():
    dtrain, dvalidate, dtest = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30 ,10])
    res = net.SGD(dtrain, epochs=50, mini_batch_size=10, eta=0.3, lmbda=0.,
            evaluation_data=dvalidate,
            monitor_freq=5,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
    with open('testdata/test_no_regu_epoch50_freq5.json', 'w') as f:
        json.dump(obj=res, fp=f)


if __name__ == '__main__':
    test_no_regu()
