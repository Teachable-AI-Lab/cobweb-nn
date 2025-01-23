# define several helper functions for cobweb-nn experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import clear_output

# define the model output class
class ModelOutput:
    def __init__(self,
                 loss: float=None,
                 centroids: list=None,
                 layer_outputs: list=None,
                 reconstructions: list=None,
                 x: torch.Tensor=None,
                logits: torch.Tensor=None,
                debug_info: dict=None):
        
        self.loss = loss
        self.centroids = centroids
        self.layer_outputs = layer_outputs
        self.reconstructions = reconstructions
        self.x = x
        self.logits = logits
        self.debug_info = debug_info

# define the model
def filter_by_label(mnist_data, labels_to_filter):
    filtered_data = []
    for data, label in tqdm(mnist_data):
        if label in labels_to_filter:
            filtered_data.append((data, label))
    return filtered_data

def visualize_centroids(centroids, layers=None, do_t_sne=False, do_pca=False):
    pass

def visualize_decision_boundary(model, val_data, layer=0, n_hidden=784):
    decisions = []
    test_examples = []
    test_targets = []
    test_representations = []

    test_loader = torch.utils.data.DataLoader(val_data, batch_size=512, shuffle=True)

    for i, (d, t) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            outputs = model(d.to('cuda'), t.to('cuda'))
            test_examples.append(d)
            test_targets.append(t)
            # print(rep.shape)
            test_representations.append(outputs.x.detach().cpu())
            # print(outputs[0])
            # print(outputs[0].argmax(dim=-1).tolist())
            # break
            # print(outputs[0].shape)
            # print(outputs[2])
            decisions.extend(outputs.layer_outputs[layer].argmax(dim=-1).tolist())
            if len(decisions) > 1000:
                break

    # plot the data, color by the decision
    # do tsne first on the data
    tsne = TSNE(n_components=2)
    tsne_outputs = tsne.fit_transform(torch.cat(test_representations).view(-1, n_hidden).numpy())
    tsne_labels = torch.cat(test_targets).numpy()
    plt.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=decisions, cmap='tab10')
    plt.colorbar()

    # create a new plot for the decision
    plt.figure()
    plt.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=tsne_labels, cmap='tab10')
    plt.colorbar()


def visualize_filters(model, layers=None):
    pass

def train_model(model, train_data=None, supervised=False, optimizer=None, device='cuda',
                batch_size=32, epochs=1,
                show_loss=False, show_centroids=False,
                show_filters=False, show_decision_boundary=False, verbose=True, early_break=False):
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    losses = []

    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for x, y in t:
                optimizer.zero_grad()

                if supervised:
                    outputs = model(x.to(device), y.to(device))
                else:
                    outputs = model(x.to(device))

                loss = outputs.loss
                losses.append(loss.item())

                if verbose:
                    print(f"layer outputs: {outputs.layer_outputs}")
                
                loss.backward()
                if early_break:
                    break 
                optimizer.step()

                t.set_postfix(loss=loss.item())


    if show_loss:
        plt.plot(losses)
        plt.show()    


def test_model(model, test_data, device='cuda', batch_size=32):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    labels = []
    pred = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x.to(device), y.to(device))
            labels.extend(y.tolist())
            pred.extend(outputs.logits.argmax(dim=-1).tolist())

        # calculate the accuracy
        correct = 0
        for l, p in zip(labels, pred):
            if l == p:
                correct += 1
        accuracy = correct / len(labels)
        print(f"Accuracy: {accuracy}")


