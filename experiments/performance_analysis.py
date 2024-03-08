import numpy as np
from generate_cell_data import generate_cell_data, get_cell_kwargs
from generate_spline_data import generate_spline_data, get_spline_kwargs
from multi_separator import GreedySeparatorShrinking, GreedySeparatorGrowing3D
from offsets import sphere_offsets, sparse_sphere_offsets
from line_interactions import get_line_costs
from partition_comparison import VariationOfInformation
from time import time
import matplotlib.pyplot as plt


def test_gss(size, seed):

    kwargs = get_cell_kwargs(0.5)

    num_vox_per_cell = 64**3 / 25
    num_cells = max(2, int(size**3 / num_vox_per_cell))

    kwargs["size"] = size
    kwargs["num_cells"] = num_cells

    r = 5
    offsets = np.concatenate([np.eye(3, dtype=int), sparse_sphere_offsets((r,) * 3, connectivity=26)])

    img, gt = generate_cell_data(seed=seed, **kwargs)
    img = np.clip(img, 1e-6, 1 - 1e-6)

    t_0 = time()
    vertex_costs = 1 - img
    interaction_costs = get_line_costs(offsets, img, np.max)
    vertex_costs = np.log(vertex_costs / (1 - vertex_costs))
    interaction_costs = np.log((1 - interaction_costs) / interaction_costs)
    t_1 = time()
    gss = GreedySeparatorShrinking()
    gss.setup_grid(vertex_costs.shape, offsets.flatten(), vertex_costs.flatten(), interaction_costs.flatten())
    del interaction_costs
    t_2 = time()
    gss.run()
    t_3 = time()

    seg = np.array(gss.vertex_labels()).reshape(vertex_costs.shape).astype(np.uint16)
    voi = VariationOfInformation(gt, seg)
    t_4 = time()
    print(f"voi = {voi.value():.3f}, fc = {voi.valueFalseCut():.3f}, fj = {voi.valueFalseJoin():.3f}")
    print(f"t_gen = {t_1 - t_0:.3f}, t_init = {t_2 - t_1:.3f}, t_run = {t_3 - t_2:.3f}, t_voi = {t_4 - t_3:.3f}")

    return t_3 - t_1, size ** 3


def test_gsg(size, seed):

    num_vox_per_spline = 64**2 / 15
    num_splines = max(2, int(np.rint(size**2 / num_vox_per_spline)))
    kwargs = get_spline_kwargs(0.5, size)
    kwargs["number_of_splines"] = num_splines
    kwargs["max_num_tries_per_spline"] = 1e4

    offsets = np.concatenate([np.eye(3, dtype=int), sphere_offsets((8,) * 3)])

    t_0 = time()
    img, gt = generate_spline_data(seed=seed, **kwargs)
    t_1 = time()

    img = np.clip(img, 1e-6, 1 - 1e-6)
    vertex_costs = np.log(img / (1 - img))

    shape = img.shape
    flattened_costs = vertex_costs.flatten(order="F")

    gsg = GreedySeparatorGrowing3D(shape, offsets, flattened_costs)
    t_2 = time()
    gsg.run()
    t_3 = time()

    vertex_labels = np.array(gsg.vertex_labels())
    vertex_labels = vertex_labels.reshape(shape, order="F")
    seg = np.require(vertex_labels, requirements="C").astype(np.uint16)
    t_4 = time()
    voi = VariationOfInformation(gt, seg)
    t_5 = time()

    print(f"voi = {voi.value():.3f}, fc = {voi.valueFalseCut():.3f}, fj = {voi.valueFalseJoin():.3f}")
    print(f"t_gen = {t_1 - t_0:.3f}, t_setup = {t_2-t_1:.3f}, t_run = {t_3 - t_2:.3f}, t_extrac = {t_4-t_3:.3f}, "
          f"t_voi = {t_5 - t_4:.3f}")

    print("num_sep =", np.count_nonzero(seg == 0))

    return t_3 - t_2, size ** 3


def evaluate(algorithm=test_gss):

    sizes = np.arange(20, 217)
    np.random.shuffle(sizes)

    num_nodes = []
    time_per_node = []

    for i, size in enumerate(sizes):
        print(f"size = {size} ({i+1}/{len(sizes)})")
        t, num_node = algorithm(size, seed=1)
        time_per_node.append(t / num_node)
        num_nodes.append(num_node)

    plot_results(num_nodes, time_per_node, algorithm.__name__[-3:])


def plot_results(num_nodes, time_per_node, name: str):

    idx = np.argsort(num_nodes)
    num_nodes = np.array(num_nodes)[idx]
    time_per_node = np.array(time_per_node)[idx]

    coefficients = np.vstack([np.log(num_nodes), np.ones(len(num_nodes))]).T

    m, c = np.linalg.lstsq(coefficients, time_per_node, rcond=None)[0]
    x = [num_nodes[0], num_nodes[-1]]
    y = [c + m * np.log(num_nodes[0]), c + m * np.log(num_nodes[-1])]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.12, left=0.12, right=0.98, top=0.95)

    ax.scatter(num_nodes, time_per_node)
    ax.plot(x, y)
    ax.set_ylabel(r"$t[s]/n$")
    ax.set_xlabel(r"$n$")
    ax.set_xscale("log")
    ax.xaxis.grid()

    plt.show()


def main():
    evaluate(test_gss)
    evaluate(test_gsg)


if __name__ == "__main__":
    main()
