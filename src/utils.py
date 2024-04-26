import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_results(res):
    epochs = np.array([*range(len(res['train_loss']))])
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, res['train_loss'], label='train loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, res['train_score'], label='train acc')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, res['valid_loss'], label='valid loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, res['valid_score'], label='valid acc')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/training_results_latest.png")
