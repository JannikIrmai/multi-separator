import numpy as np
from skimage.segmentation import watershed
from skimage.morphology import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import pyvista as pv


def seeded_region_growing(image: np.ndarray, t_min: float = None, t_max: float = None, watershed_line: bool = False):
    markers = None if t_min is None else label(image <= t_min, connectivity=6)
    mask = None if t_max is None else image <= t_max
    labels = watershed(image, markers=markers, mask=mask, watershed_line=watershed_line).astype(np.uint16)
    return labels


def plot_quartile_curve(values, t, ax, color=None, name: str = None):

    median = np.median(values, axis=1)
    quart_1 = np.quantile(values, 1/4, axis=1)
    quart_3 = np.quantile(values, 3/4, axis=1)

    min_val = np.quantile(values, 1 / 10, axis=1)
    max_val = np.quantile(values, 9 / 10, axis=1)

    ax.plot(t, median, color=color, label=name)
    ax.fill_between(t, quart_1, quart_3, fc=color, alpha=0.3, ec=None)
    ax.plot(t, min_val, color=color, ls="dashed", lw=0.5)
    ax.plot(t, max_val, color=color, ls="dashed", lw=0.5)


def match_labels(gt: np.ndarray, pred: np.ndarray):
    data = []
    row = []
    col = []
    labels = np.unique(gt)
    labels = labels[labels > 0]
    max_pred = np.max(pred)
    for i, lab in enumerate(labels):
        matches, counts = np.unique(pred[gt==lab], return_counts=True)
        counts = counts[matches > 0]
        matches = matches[matches > 0]
        data += list(counts)
        row += [i] * len(counts)
        col += list(matches)
        # add dummy label
        data.append(1)
        row.append(i)
        col.append(max_pred + i + 1)
    mat = csr_matrix((data, (row, col)), dtype=int)

    row_ind, col_ind = min_weight_full_bipartite_matching(mat, maximize=True)

    for row, lab in zip(row_ind, col_ind):
        if labels[row] == lab:
            continue
        tmp = pred == lab
        pred[pred == labels[row]] = lab
        pred[tmp] = labels[row]
        col_ind[col_ind == labels[row]] = lab


def plot_cube(img: np.ndarray, file_name: str = "results/cube.png", **kwargs):
    p = pv.Plotter()
    p.set_background("white")
    grid = pv.UniformGrid()
    shape = np.array(img.shape)
    grid.dimensions = shape + 1
    grid.cell_data["values"] = img.flatten(order="F")
    p.add_mesh(grid, show_scalar_bar=False, **kwargs)
    p.show(screenshot=file_name, interactive=False, window_size=(1024, 1024))


def plot_voxels(img: np.ndarray, file_name: str = "results/cube.png", **kwargs):
    p = pv.Plotter()
    p.set_background("white")
    grid = pv.UniformGrid()
    shape = np.array(img.shape)
    grid.dimensions = shape + 1
    grid.cell_data["values"] = img.flatten(order="F")
    cells = grid.extract_cells(img.flatten("F") > 0)
    p.add_mesh(cells, show_scalar_bar=False, **kwargs)
    p.show(screenshot=file_name, interactive=False, window_size=(1024, 1024))
