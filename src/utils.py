import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(res):
    epochs = np.array([*range(len(res['loss']))])
    plt.plot(epochs, res['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
