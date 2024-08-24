from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualise_tsne(features, labels, ax):
    z = TSNE(n_components=2).fit_transform(features.detach().cpu().numpy())
    ax.scatter(z[:, 0], z[:, 1], s=50, c=labels)
    ax.axis("off")

def visualize_output(h, color, ax):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    ax.scatter(z[:, 0], z[:, 1], s=50, c=color, cmap="Set2")
    ax.axis('off')

def perturbed_tsne(data, data_p):
    fig, axs = plt.subplots(1, 2,figsize=(20, 10))
    axs[0].set_title('t-SNE visualisation of original features.', fontsize=10)
    visualise_tsne(data.x, data.y, axs[0])
    axs[1].set_title('t-SNE visualisation of randomly perturbed data features.', fontsize=10)
    visualise_tsne(data_p.x, data_p.y, axs[1])