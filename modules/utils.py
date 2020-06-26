import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, MetaEstimatorMixin
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib as mpl
from scipy.stats import norm
from scipy import linalg
from matplotlib.colors import ListedColormap
import pandas as pd 

def add_decision_boundary(model, levels=None, resolution=1000, ax=None, label=None, color=None):
    """Trace une frontière de décision sur une figure existante.
                    
    La fonction utilise `model` pour prédire un score ou une classe
    sur une grille de taille `resolution`x`resolution`. Une (ou
    plusieurs frontières) sont ensuite tracées d'après le paramètre
    `levels` qui fixe la valeur des lignes de niveaux recherchées.
    """
    if ax is None:
        ax = plt.gca()
    if callable(model):
        if levels is None:
            levels = [0]
        def predict(X):
            return model(X)
    else:
        n_classes = len(model.classes_)
        if n_classes == 2:
            if hasattr(model, "decision_function"):
                if levels is None:
                    levels = [0]
                def predict(X):
                    return model.decision_function(X)
            elif hasattr(model, "predict_proba"):
                if levels is None:
                    levels = [.5]
                def predict(X):
                    pred = model.predict_proba(X)
                    if pred.shape[1] > 1:
                        return pred[:, 0]
                    else:
                        return pred
            elif hasattr(model, "predict"):
                if levels is None:
                    levels = [.5]
                def predict(X):
                    return model.predict(X)
            else:
                raise Exception("Modèle pas reconnu")
        else:
            levels = np.arange(n_classes - 1) + .5
            def predict(X):
                pred = model.predict(X)
                _, idxs = np.unique(pred, return_inverse=True)
                return idxs
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = predict(xy).reshape(XX.shape)
    color = "red" if color is None else color
    sns.lineplot([0], [0], label=label, ax=ax, color=color, linestyle="dashed")
    ax.contour(
        XX,
        YY,
        Z,
        levels=levels,
        colors=[color],
        linestyles="dashed",
        antialiased=True,
    ) 
       
def better_decision_boundary(clf, X, y, labels, X_test, y_test):

    h = 5  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['tab:green','tab:orange' , 'tab:blue'])
    cmap_bold = ListedColormap(['darkgreen', 'darkorange' , 'blue' ])

    # we create an instance of Neighbours Classifier and fit the data.
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    X_concat = np.append(X, X_test, axis=0)
    x_min, x_max = X_concat[:, 0].min() - 100, X_concat[:, 0].max() + 100
    y_min, y_max = X_concat[:, 1].min() - 100, X_concat[:, 1].max() + 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15,10))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    cmap_test = np.array(pd.factorize(y_test)[1])
    cmap_test[cmap_test=='Normal'] = 'blue'
    cmap_test[cmap_test=='Hernia'] = 'darkgreen'
    cmap_test[cmap_test=='Spondylolisthesis'] = 'darkorange'
    cmap_test = ListedColormap(cmap_test.tolist())
    scatter_test = plt.scatter(X_test[:,0], X_test[:,1], c=pd.factorize(y_test)[0],  label=labels, cmap=cmap_test, marker='^', edgecolor='k')
    legend_test = plt.legend(*scatter_test.legend_elements(), loc=(0.81,0.78))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20, label=labels)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    legend = plt.legend(*scatter.legend_elements(), loc="upper right" , title="Class")
    
    for i, label in enumerate(labels):
        legend.get_texts()[i].set_text(f'{label} (train)')

    for i, label in enumerate(pd.factorize(y_test)[1]):
        legend_test.get_texts()[i].set_text(f'{label} (test)')
    plt.gca().add_artist(legend)
    plt.gca().add_artist(legend_test)
    
def plot_clustering(data, labels, markers=None, ax=None, **kwargs):
    """Affiche dans leur premier plan principal les données `data`,
colorée par `labels` avec éventuellement des symboles `markers`.
    """

    if ax is None:
        ax = plt.gca()

    # Reduce to two dimensions
    if data.shape[1] == 2:
        data_pca = data.to_numpy()
    else:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

    COLORS = np.array(['blue', 'green', 'red', 'purple', 'gray', 'cyan'])
    _, labels = np.unique(labels, return_inverse=True)
    colors = COLORS[labels]

    if markers is None:
        ax.scatter(*data_pca.T, c=colors)
    else:
        MARKERS = "o^sP*+xD"

        # Use integers
        markers_uniq, markers = np.unique(markers, return_inverse=True)

        for marker in range(len(markers_uniq)):
            data_pca_marker = data_pca[markers == marker, :]
            colors_marker = colors[markers == marker]
            ax.scatter(*data_pca_marker.T, c=colors_marker, marker=MARKERS[marker])

    if 'centers' in kwargs and 'covars' in kwargs:
        if data.shape[1] == 2:
            centers_2D = kwargs['centers']
            covars_2D = kwargs['covars']
        else:
            centers_2D = pca.transform(kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T
                for c in kwargs['covars']
            ]

        p = 0.9
        sig = norm.ppf(p**(1/2))

        for i, (covar_2D, center_2D) in enumerate(zip(covars_2D, centers_2D)):
            v, w = linalg.eigh(covar_2D)
            print(v)
            v = 2. * sig * np.sqrt(v)

            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])

            color = COLORS[i]
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    return ax



