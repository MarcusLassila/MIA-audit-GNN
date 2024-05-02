import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import Dataset

class AttackDataset(Dataset):
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def plot_training_results(res, name, savedir):
    epochs = np.array([*range(len(res['train_loss']))])
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, res['train_loss'], label='train loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, res['train_score'], label='train score')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, res['valid_loss'], label='valid loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, res['valid_score'], label='valid score')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.grid(True)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{savedir}/training_results_{name}.png")
    plt.clf()

def plot_roc_loglog(fpr, tpr, name, savedir='plots'):
    plt.figure(figsize=(8, 8))
    plt.loglog(fpr, tpr)
    plt.xlim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{savedir}/{name}_roc_loglog.png")
    plt.clf()
