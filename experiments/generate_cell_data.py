import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, segmentation
from synth_growth import synth_growth_2d, synth_growth_3d
from scipy.ndimage import distance_transform_edt


def smooth_separator(labels):
    shape = labels.shape
    # remove separator holes
    while True:
        num_separator_neighbors = np.zeros(shape, dtype=np.uint8)
        for d in range(3):
            for i in [-1, +1]:
                slicer_a = tuple([slice(None) if j != d else slice(max(0, i), shape[j] + min(0, i)) for j in range(3)])
                slicer_b = tuple(
                    [slice(None) if j != d else slice(max(0, -i), shape[j] + min(0, -i)) for j in range(3)])
                num_separator_neighbors[slicer_a] += labels[slicer_b] == 0
        num_separator_neighbors[labels == 0] = 0
        mask = num_separator_neighbors > 3
        if np.count_nonzero(mask) == 0:
            break
        labels[mask] = 0

    # remove separator spikes
    while True:
        neighbors = np.zeros((6,) + shape, labels.dtype)
        counter = 0
        for d in range(3):
            for i in [-1, +1]:
                slicer_a = tuple([slice(None) if j != d else slice(max(0, i), shape[j] + min(0, i)) for j in range(3)])
                slicer_b = tuple(
                    [slice(None) if j != d else slice(max(0, -i), shape[j] + min(0, -i)) for j in range(3)])
                neighbors[counter][slicer_a] = labels[slicer_b]
                counter += 1

        num_neighbors = np.count_nonzero(neighbors != 0, axis=0)
        comp = np.max(neighbors, axis=0)
        mask = (labels == 0) & (num_neighbors >= 5) & np.all((neighbors == comp) | (neighbors == 0), axis=0)

        if np.count_nonzero(mask) == 0:
            break

        labels[mask] = comp[mask]


def three_components_touching(labels):
    dim = labels.ndim

    shape = labels.shape
    neighbors = np.empty((1 + 2*dim,) + shape, dtype=labels.dtype)
    neighbors[:] = labels
    counter = 0
    for d in range(dim):
        for i in [-1, 1]:
            slicer_a = tuple([slice(None) if j != d else slice(max(i, 0), shape[j] + min(i, 0)) for j in range(dim)])
            slicer_b = tuple([slice(None) if j != d else slice(max(-i, 0), shape[j] + min(-i, 0)) for j in range(dim)])
            counter += 1
            neighbors[counter][slicer_a] = labels[slicer_b]
    neighbors.sort(axis=0)
    diff = np.diff(neighbors, axis=0) > 0
    num_unique_neighbors = np.sum(diff, axis=0)
    three_touching = num_unique_neighbors >= 2
    return three_touching


def generate_cell_data(size: int = 64, seed: int = 1,
                       num_cells: int = 10, membrane_width: float = 3., min_cell_diameter: int = 8,
                       node_distance: float = 0, node_factor: float = 1,
                       mean_membrane: float = 0.7, std_membrane: float = 0.2, mean_cell: float = 0.2,
                       std_cell: float = 0.1, max_p_membrane: float = 0.9, mixture: bool = True, dim=3):
    assert membrane_width >= 1

    # generate random multi separator by randomly growing num_cells components until their pixel boundaries bump into
    # each other
    synth_growth = synth_growth_2d if dim == 2 else synth_growth_3d
    labels = np.array(synth_growth(size, num_cells, seed), dtype=np.uint16).reshape((size,) * dim)

    # erode the components such that the components that are connected by a bottleneck that is smaller
    # than min_cell_diameter are split into two components
    radius = int((min_cell_diameter + membrane_width) / 2)
    mesh = np.array(np.meshgrid(*[np.arange(-radius, radius + 1) for _ in range(dim)]))
    kernel = np.sum(mesh ** 2, axis=0) <= radius ** 2
    labels = morphology.erosion(labels, kernel)

    # only keep the num_cells largest components
    labels = morphology.label(labels, connectivity=1)
    index, count = np.unique(labels, return_counts=True)
    count = count[index != 0]
    index = index[index != 0]
    index = index[np.argsort(-count)]
    for i in index[num_cells:]:
        labels[labels == i] = 0

    # expand the eroded labels until the components bump into each other
    labels = segmentation.expand_labels(labels, distance=size)

    # compute all points where three components touch
    if node_distance == 0 or node_factor == 0:
        dist_node = np.ones(labels.shape)
    else:
        three_touching = three_components_touching(labels)
        dist_node = distance_transform_edt(~three_touching)
        dist_node = np.clip(node_distance - dist_node, 0, node_distance) / node_distance
        dist_node *= node_factor
        dist_node += 1

    # insert a separator between all components
    separator = np.zeros(labels.shape, dtype=bool)
    for i in range(dim):
        slicer_a = tuple([slice(None) if j != i else slice(1, None) for j in range(dim)])
        slicer_b = tuple([slice(None) if j != i else slice(0, -1) for j in range(dim)])
        mask = (labels[slicer_a] != labels[slicer_b]) & (labels[slicer_b] != 0) & (labels[slicer_a] != 0)
        separator[slicer_a][mask] = True

    # compute the distance transform
    dist = distance_transform_edt(~separator)
    from skimage.filters import gaussian
    dist = gaussian(dist, 0.5)
    dist += 1/2

    # amplify the distance around the nodes
    dist /= dist_node

    dist[separator & (dist > membrane_width / 2)] = membrane_width / 2

    # compute the ground truth labeling
    labels[dist <= membrane_width/2] = 0

    # sample a gray value image from a mixture normal distributions.
    gt = morphology.label(labels, connectivity=1)

    if max_p_membrane < 1:
        theta_1 = -np.log(1 - max_p_membrane) + np.log(max_p_membrane)
        theta_2 = - theta_1 / (membrane_width / 2)
        membrane_weight = 1 / (1 + np.exp(-theta_1-theta_2*dist))
    else:
        membrane_weight = (dist <= membrane_width / 2).astype(float)

    np.random.seed(seed)
    cell_sample = np.random.normal(mean_cell, std_cell, size=labels.shape)
    membrane_sample = np.random.normal(mean_membrane, std_membrane, size=labels.shape)
    if mixture:
        image = np.where(np.random.random(labels.shape) < membrane_weight, membrane_sample, cell_sample)
    else:
        image = membrane_weight * membrane_sample + (1 - membrane_weight) * cell_sample
    image = np.clip(image, 0, 1)

    return image, gt.astype(np.uint16)


def get_cell_kwargs(t):
    size = 64

    mm = [0.7, 0.55]
    mc = [0.3, 0.45]
    sm = [0.05, 0.1]
    sc = [0.05, 0.1]

    kwargs = {
        "size": size,
        "num_cells": 25,
        "membrane_width": 1.5,
        "node_distance": 10,
        "node_factor": 2,
        "min_cell_diameter": 8,
        "mean_membrane": (1 - t) * mm[0] + t * mm[1],
        "std_membrane": (1 - t) * sm[0] + t * sm[1],
        "mean_cell": (1 - t) * mc[0] + t * mc[1],
        "std_cell": (1 - t) * sc[0] + t * sc[1],
        "max_p_membrane": 0.9,  # max(0.5 + epsilon, min(1 - t / 2, 1 - epsilon))
        "mixture": False
    }
    return kwargs


def test_2d():
    img, labels = generate_cell_data(size=100, dim=2, mixture=False, seed=1)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(labels, cmap="tab20")
    plt.show()


if __name__ == "__main__":
    test_2d()
