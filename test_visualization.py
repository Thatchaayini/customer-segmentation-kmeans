import numpy as np
import pandas as pd

from visualization import reduce_pca, reduce_tsne, plot_clusters


def make_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))


def test_pca_returns_two_columns():
    coords, variance = reduce_pca(make_data())
    assert coords.shape == (40, 2)
    assert len(variance) == 2


def test_pca_variance_is_valid():
    _, variance = reduce_pca(make_data())
    assert (variance > 0).all()
    assert variance.sum() <= 1.0


def test_tsne_returns_two_columns():
    coords = reduce_tsne(make_data())
    assert coords.shape == (40, 2)


def test_tsne_handles_small_data():
    small = make_data().head(5)
    coords = reduce_tsne(small)
    assert coords.shape == (5, 2)


def test_plot_saves_image(tmp_path):
    coords, _ = reduce_pca(make_data())
    labels = np.array([0, 1] * 20)
    path = str(tmp_path / "charts" / "plot.png")
    plot_clusters(coords, labels, "Test", "x", "y", path)
    assert (tmp_path / "charts" / "plot.png").exists()