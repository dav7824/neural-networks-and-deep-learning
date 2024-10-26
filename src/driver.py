import sys
import json
import matplotlib.pyplot as plt
import mnist_loader
import network
import network_mat_based
import network2
import network2_L1L2


def test_mat_based():
    print('===Test matrix-based BPG===')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network_mat_based.Network([784, 30, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, test_data=test_data)
    print('\n\n')

def test_regularisation(hidden_layers, epochs, mini_batch_size, eta, lmbda, select_reg):
    if select_reg == 'L1':
        reg = network2_L1L2.L1_Regularisation
    elif select_reg == 'L2':
        reg = network2_L1L2.L2_Regularisation
    else:
        sys.exit('[ERROR] Invalid regularisation setting')
    # read dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # create neural net
    layers = [784] + hidden_layers + [10]
    net = network2_L1L2.Network([784, 30, 10], reg=reg)
    # run SGD
    return net.SGD(training_data, epochs, mini_batch_size, eta, lmbda, 
            evaluation_data=validation_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
            )
    print('\n\n')

def run_L1L2_test():
    resL2 = test_regularisation(hidden_layers=[30], epochs=30, mini_batch_size=10, 
                                eta=0.3, lmbda=5., select_reg='L2')
    resL1 = test_regularisation(hidden_layers=[30], epochs=30, mini_batch_size=10, 
                                eta=0.3, lmbda=5., select_reg='L1')
    with open('testdata/L1L2_test.json', 'w') as f:
        json.dump({'L2': resL2, 'L1': resL1}, f, indent=4)

def plot_L1L2_eval_acc():
    with open('testdata/L1L2_test.json') as f:
        res = json.load(f)
    fig, ax = plt.subplots()
    ax.plot(res['L2']['eval_acc'], label='L2')
    ax.plot(res['L1']['eval_acc'], label='L1')
    ax.legend()
    plt.savefig('testdata/L1L2_eval_acc.png')
    plt.close()

def plot_overfit():
    with open('testdata/L1L2_test.json') as f:
        res = json.load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(res['L1']['train_cost'], label='Training cost')
    ax1.plot(res['L1']['eval_cost'], label='Validation cost')
    ax1.set_title('L1')
    ax1.legend()
    ax2.plot(res['L2']['train_cost'], label='Training cost')
    ax2.plot(res['L2']['eval_cost'], label='Validation cost')
    ax2.set_title('L2')
    ax2.legend()
    plt.savefig('testdata/L1_L2_overfit.png')
    plt.close()


if __name__ == '__main__':
    plot_overfit()
