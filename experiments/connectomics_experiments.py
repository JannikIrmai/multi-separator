import cc3d
import h5py
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from color import *
import pickle
import os
from time import time
from partition_comparison import VariationOfInformation
import multi_separator
import multi_cut


connectomics_data_filename = "data/connectomics-data.h5"

biases = {
    "chunk_wise": np.round(np.linspace(-0.7, -0.3, 9), 3),
    "multicut": np.round(np.linspace(-0.5, 0.1, 13), 3),
    "multi_separator": np.round(np.linspace(-0.7, -0.1, 13), 3)
}


def ms_mutex(membranes: np.ndarray, affinities: np.ndarray, offsets: np.ndarray, bias: float):
    shape = membranes.shape
    weights = np.concatenate([np.expand_dims((1 - bias) - membranes, 0), affinities])
    weights = np.moveaxis(weights, 0, -1)
    grid_mutex = multi_separator.GridMutex3D(shape, offsets)

    # sort the array indices
    sorted_indices = np.argsort(weights.ravel())
    grid_mutex.run(sorted_indices)

    # ignore all components of size 1
    vertex_labels = np.array(grid_mutex.vertex_labels(1), dtype=np.uint16).reshape(shape)

    return vertex_labels


def get_foreground_mask(affinities, membranes, offsets, theta):
    """
    code from https://github.com/markschoene/MeLeCoLe/blob/main/melecole/infer.py
    """
    mask = np.ones_like(affinities, dtype=bool)

    # get largest offset number
    pad_size = np.max(np.abs(np.array(offsets)))

    # initialize padded foreground
    foreground = np.pad(membranes > theta, pad_width=pad_size, mode='constant', constant_values=1).astype(bool)

    # compute foreground mask for each offset vector
    for i, vector in enumerate(offsets):
        dims = membranes.shape
        slices_null = [slice(pad_size, pad_size + dims[k]) for k in range(len(dims))]
        slices_plus = [slice(pad_size + vector[k], pad_size + vector[k] + dims[k]) for k in range(len(dims))]

        # remove both edges that are associated with pixel (i, j, k)
        # that is (offset_1, offset_2, offset_3) + (i, j, k) AND (i, j, k)
        mask[i] = np.logical_and(foreground[slices_plus[0], slices_plus[1], slices_plus[2]],
                                 foreground[slices_null[0], slices_null[1], slices_null[2]])
    return mask


def mc_mutex(membranes: np.ndarray, affinities: np.ndarray, offsets: np.ndarray, bias: float,
             background_threshold: float = 0.6):

    foreground_mask = get_foreground_mask(affinities, membranes, offsets, background_threshold)
    affinities = affinities.copy()
    affinities[3:] *= -1
    affinities[3:] += 1
    affinities[:3] += bias

    affinities = np.moveaxis(affinities, 0, -1)
    foreground_mask = np.moveaxis(foreground_mask, 0, -1)
    affinities = np.require(affinities, requirements='C')
    foreground_mask = np.require(foreground_mask, requirements='C')

    vol_shape = membranes.shape
    mutex = multi_cut.GridMutex3D(vol_shape, offsets, 3)

    # sort in descending order
    sorted_edges = np.argsort(-affinities.ravel())
    # remove edges adjacent to background voxels from graph
    sorted_edges = sorted_edges[foreground_mask.ravel()[sorted_edges]]
    # run the mutex watershed
    mutex.run(sorted_edges)

    segmentation = np.array(mutex.vertex_labels(1), dtype=np.uint16).reshape(vol_shape)
    return segmentation


def chunk_wise_ms(membranes, affinities, offsets, chunk_size, step_size, padding, bias_cut):
    shape = membranes.shape

    assert all(2*padding[i] < shape[i] for i in range(len(shape)))

    num_steps = [int(np.ceil(max(0, shape[i] - 2*padding[i] - chunk_size[i]) / step_size[i])) + 1
                 for i in range(len(shape))]
    slicers = []
    pad_slicers = []
    for step in product(*[range(n) for n in num_steps]):
        slicer = tuple([slice(step_size[i] * step[i],
                              min(shape[i], step_size[i] * step[i] + chunk_size[i] + 2*padding[i]))
                        for i in range(len(shape))])
        pad_slicers.append(slicer)
        slicer = tuple([slice(step_size[i] * step[i] + padding[i],
                              min(shape[i]-padding[i], step_size[i] * step[i] + chunk_size[i] + padding[i]))
                        for i in range(len(shape))])
        slicers.append(slicer)

    separator = np.zeros(shape, dtype=int)

    for i in range(len(slicers)):
        print("\rProcessing chunk", i+1, "/", len(slicers), end="")
        chunk_membranes = membranes[pad_slicers[i]]
        chunk_affinities = affinities[(slice(None),) + pad_slicers[i]]
        chunk_vertex_labels = ms_mutex(chunk_membranes, chunk_affinities, offsets, bias_cut=bias_cut)

        padding_slicer = tuple(slice(padding[k], chunk_vertex_labels.shape[k] - padding[k]) for k in range(len(shape)))
        separator[slicers[i]] += chunk_vertex_labels[padding_slicer] == 0
    print("\r" + " "*30)

    return separator


def compute_chunk_wise_segmentations():

    ms_padding = (2, 4, 4)
    chunk_size = (10, 32, 32)
    step_size = (5, 16, 16)

    file = h5py.File(connectomics_data_filename, "r")
    offsets = file["offsets"][:]
    padding = file["padding"][:]
    cost_shape = file['membranes'].shape

    cost_slicer = tuple(slice(padding[i] - ms_padding[i], cost_shape[i] - padding[i] + ms_padding[i])
                        for i in range(3))

    membranes = file["membranes"][cost_slicer]
    affinities = file["affinities"][(slice(None),) + cost_slicer]
    file.close()

    for bias in biases["chunk_wise"]:
        result_file_name = f"results/connectomics/chunk_wise_{str(bias).replace('.', '')}.h5"
        if os.path.isfile(result_file_name):
            print(f"Chunk_wise mutex segmentation for bias {bias} already exists.")
            continue

        print("Chunk_wise multi-separator for bias = ", bias)

        t = time()
        separator = chunk_wise_ms(membranes, affinities, offsets, chunk_size, step_size,
                                  padding=ms_padding, bias_cut=bias)
        print("Time:", time() - t)

        shape = separator.shape
        padding_slicer = tuple(slice(ms_padding[k], shape[k] - ms_padding[k]) for k in range(len(shape)))
        separator = separator[padding_slicer]

        seg = cc3d.connected_components(separator == 0, connectivity=6).astype(np.uint16)

        file = h5py.File(result_file_name, "w-")
        file.create_dataset("segmentation", data=seg)
        file.close()


def compute_segmentations(method):

    file = h5py.File(connectomics_data_filename, "r")
    offsets = file["offsets"][:]
    padding = file["padding"][:]
    cost_shape = file['membranes'].shape
    cost_slicer = tuple(slice(padding[i], cost_shape[i] - padding[i])
                        for i in range(3))
    membranes = file["membranes"][cost_slicer]
    affinities = file["affinities"][(slice(None),) + cost_slicer]
    file.close()

    for bias in biases[method]:
        result_file_name = f"results/connectomics/{method}_{str(bias).replace('.', '')}.h5"
        if os.path.isfile(result_file_name):
            print(f"{method} segmentation for bias {bias} already exists.")
            continue

        print(f"{method} for bias = {bias}")
        t = time()
        if method == "multicut":
            segmentation = mc_mutex(membranes, affinities, offsets, bias=bias)
        elif method == "multi-separator":
            segmentation = ms_mutex(membranes, affinities, offsets, bias=bias)
        print("Time:", time() - t)

        file = h5py.File(result_file_name, 'w-')
        file.create_dataset('segmentation', data=segmentation)
        file.close()


def evaluate(method="chunk_wise"):

    # load ground truth
    file = h5py.File(connectomics_data_filename, "r")
    labels = file["processed_labels"][:]
    file.close()

    vois = []

    for bias in biases:
        filename = f"results/connectomics/{method}_{str(bias).replace('.', '')}.h5"
        # load results
        file = h5py.File(filename)
        segmentation = file["segmentation"][:]
        file.close()

        # compute VoI
        voi = VariationOfInformation(segmentation, labels, True)
        fj, fc = voi.valueFalseJoin(), voi.valueFalseCut()
        vois.append((fj, fc))
        print(f"{method}, bias={bias}, VI={fj+fc:.3f}, FC={fc:.3f}, FJ={fj:.3f}")

    vois = np.array(vois)

    with open(f"results/connectomics/{method}_voi.pickle", "wb") as f:
        pickle.dump({"biases": biases, "vois": vois}, f)


def plot_results():
    methods = ["multi_separator", "chunk_wise", "multicut"]
    method_names = [r"multi-separator", r"chunk-wise multi-separator", r"multicut"]
    fig, ax = plt.subplots(1, len(methods), figsize=(4*len(methods), 3), sharey=True)

    for i, method in enumerate(methods):
        with open(f"results/connectomics/{method}_voi.pickle", "rb") as f:
            results = pickle.load(f)

        ax[i].plot(-results["biases"], results["vois"][:, 0], label="FJ", color=YELLOW)
        ax[i].plot(-results["biases"], results["vois"][:, 1], label="FC", color=GREEN)
        ax[i].plot(-results["biases"], results["vois"].sum(axis=1), label="VI", color=RED)
        ax[i].set_xlabel(r"$b$")
        if i == 0:
            ax[i].legend()
        ax[i].set_title(method_names[i])
    fig.tight_layout()
    plt.show()

    fig.savefig(f"results/connectomics/voi-comparison.png", dpi=300)


def main():
    compute_segmentations("multicut")
    compute_segmentations("multi_separator")
    compute_chunk_wise_segmentations()

    evaluate("multicut")
    evaluate("multi_separator")
    evaluate("chunk_wise")

    plot_results()


if __name__ == "__main__":
    main()
