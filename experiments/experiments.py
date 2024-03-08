import numpy as np
from offsets import sphere_offsets, sparse_sphere_offsets
import h5py
from generate_spline_data import get_spline_kwargs, generate_spline_data
from generate_cell_data import get_cell_kwargs, generate_cell_data
from line_interactions import get_line_costs
from multi_separator import GreedySeparatorShrinking, GreedySeparatorGrowing3D
from partition_comparison import VariationOfInformation, WeightedSingletonVariationOfInformation
from utils import seeded_region_growing, plot_quartile_curve, plot_voxels, plot_cube
import matplotlib.pyplot as plt
from colorcet import glasbey
from matplotlib.colors import to_rgb, to_hex, ListedColormap
from utils import match_labels
from color import *


def setup_results_file(img_type="filament"):

    num_seeds = 10

    if img_type == "filament":
        offsets = np.concatenate([np.eye(3, dtype=int), sphere_offsets((8,) * 3)])
        biases = np.round(np.linspace(-0.25, 0.25, 51), 3)
        t_min_interval = (0.45, 0.65, 41)
        t_max_interval = (0.5, 0.7, 41)
    elif img_type == "cell":
        offsets = np.concatenate([np.eye(3, dtype=int), sparse_sphere_offsets((5,) * 3, connectivity=26)])
        biases = np.round(np.linspace(-0.25, 0.25, 51), 3)
        t_min_interval = (0.0, 0.5, 51)
        t_max_interval = (0.4, 0.6, 21)
    else:
        raise ValueError(f"Invalid img_type '{img_type}'")

    thresholds = []
    for t_min in np.linspace(*t_min_interval):
        for t_max in np.linspace(*t_max_interval):
            t_min = np.round(t_min, 5)
            t_max = np.round(t_max, 5)
            if t_min > t_max:
                continue
            thresholds.append((t_min, t_max))

    noise_levels = np.round(np.linspace(0, 1, 21), 3)

    file = h5py.File(f"results/{img_type}_results.h5", "w-")
    file.create_dataset("num_seeds", data=num_seeds)
    file.create_dataset("noise_levels", data=noise_levels)

    synth_img_group = file.create_group("synth_img")

    ms_group = file.create_group("multi_separator")
    ms_group.create_dataset("biases", data=biases)
    ms_group.create_dataset("offsets", data=offsets)

    ws_group = file.create_group("watershed")
    ws_group.create_dataset("thresholds", data=thresholds)

    get_kwargs = get_spline_kwargs if img_type == "filament" else get_cell_kwargs
    generate_data = generate_spline_data if img_type == "filament" else generate_cell_data

    for seed in range(1, num_seeds + 1):
        print("seed:", seed)
        seed_group = synth_img_group.create_group(f"seed_{seed}")
        seed_group_ms = ms_group.create_group(f"seed_{seed}")
        seed_group_ws = ws_group.create_group(f"seed_{seed}")
        for i, t in enumerate(noise_levels):
            kwargs = get_kwargs(t)
            img, gt = generate_data(seed=seed, **kwargs)
            if i == 0:
                seed_group.create_dataset("gt", data=gt)
            t_str = f"{t:.3f}".replace(".", "")
            seed_group.create_dataset(f"img_{t_str}", data=img)
            seed_group_ms.create_group(f"img_{t_str}")
            seed_group_ws.create_group(f"img_{t_str}")

    file.close()


def add_noise_levels(img_type, noise_levels_to_add):
    file = h5py.File(f"results/{img_type}_results.h5", "r+")
    noise_levels = file["noise_levels"][:]
    noise_levels = np.unique(np.concatenate([noise_levels, noise_levels_to_add]))
    np.sort(noise_levels)
    input(f"new noise levels = {noise_levels}. Continue?")
    del file["noise_levels"]
    file.create_dataset("noise_levels", data=noise_levels)

    num_seeds = file["num_seeds"][()]

    get_kwargs = get_spline_kwargs if img_type == "filament" else get_cell_kwargs
    generate_data = generate_spline_data if img_type == "filament" else generate_cell_data

    for seed in range(1, num_seeds + 1):
        print("seed:", seed)
        for i, t in enumerate(noise_levels):
            t_str = f"{t:.3f}".replace(".", "")
            if f"img_{t_str}" in file["synth_img"][f"seed_{seed}"]:
                continue
            print(f"t = {t}")
            kwargs = get_kwargs(t)
            img, gt = generate_data(seed=seed, **kwargs)
            file["synth_img"][f"seed_{seed}"].create_dataset(f"img_{t_str}", data=img)
            file["multi_separator"][f"seed_{seed}"].create_group(f"img_{t_str}")
            file["watershed"][f"seed_{seed}"].create_group(f"img_{t_str}")

    file.close()


def compute_segmentations(img_type, seed, t):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    ms_group = file["multi_separator"]
    seed_group = file["synth_img"][f"seed_{seed}"]
    t_str = f"{t:.3f}".replace(".", "")
    if f"seg" in file["multi_separator"][f"seed_{seed}"][f"img_{t_str}"]:
        raise FileExistsError(f"Segmentations for img_type = {img_type}, seed = {seed}, t = {t} already exist.")

    img = seed_group[f"img_{t_str}"][:]
    offsets = ms_group["offsets"][:]
    biases = ms_group["biases"][:]
    file.close()

    print(f"Running ms for img_type = {img_type}, seed = {seed}, t = {t_str}")
    ms_algo = compute_filament_segmentations_ms if img_type == "filament" else compute_cell_segmentations_ms
    segmentations = ms_algo(img, offsets, biases)

    while True:
        try:
            file = h5py.File(f"results/{img_type}_results.h5", "r+")
            group = file["multi_separator"][f"seed_{seed}"][f"img_{t_str}"]
            group.create_dataset(f"seg", data=segmentations)
            file.close()
            break
        except BlockingIOError:
            print(f"\rCannot save segmentation, file blocked!", end="", flush=True)
            pass


def compute_segmentation_all_seeds(img_type, t):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    num_seeds = file["num_seeds"][()]
    file.close()
    for seed in range(1, num_seeds + 1):
        try:
            compute_segmentations(img_type, seed, t)
        except FileExistsError:
            print(f"segmentations for img_type = {img_type}, seed = {seed}, t = {t} already exist")
            continue


def evaluate_ms_segmentation(img_type, seed, t):
    file = h5py.File(f"results/{img_type}_results.h5", "r+")
    t_str = f"{t:.3f}".replace(".", "")

    if len(file["multi_separator"][f"seed_{seed}"][f"img_{t_str}"]) > 1:
        raise FileExistsError(f"MS results for img_type={img_type}, seed={seed}, t={t_str} already exist")

    print(f"Evaluating MS for img_type = {img_type}, seed = {seed}, t = {t_str}")

    gt = file["synth_img"][f"seed_{seed}"]["gt"][:]
    ms_group = file["multi_separator"][f"seed_{seed}"][f"img_{t_str}"]
    segmentations = ms_group[f"seg"][:]

    results = {}
    for i, seg in enumerate(segmentations):
        res = evaluate_segmentation(gt, seg)
        for key, val in res.items():
            if i == 0:
                results[key] = []
            results[key].append(val)

    for key, val in results.items():
        ms_group.create_dataset(key, data=val)
    file.close()


def evaluate_watershed_segmentation(img_type, seed, t):
    t_str = f"{t:.3f}".replace(".", "")

    file = h5py.File(f"results/{img_type}_results.h5", "r")
    if len(file["watershed"][f"seed_{seed}"][f"img_{t_str}"]) > 0:
        raise FileExistsError(f"Watershed results for img_type={img_type}, seed={seed}, t={t_str} already exist")

    gt = file["synth_img"][f"seed_{seed}"]["gt"][:]
    img = file["synth_img"][f"seed_{seed}"][f"img_{t_str}"][:]
    thresholds = file["watershed"]["thresholds"][:]
    file.close()

    print(f"Evaluating watershed for img_type = {img_type}, seed = {seed}, t = {t_str}")

    results = {}
    for i, (t_min, t_max) in enumerate(thresholds):
        print(f"\rwatershed threshold {i+1} / {len(thresholds)}", end="", flush=True)
        seg = seeded_region_growing(img if img_type == "cell" else (1 - img), t_min, t_max,
                                    watershed_line=img_type == "cell")
        res = evaluate_segmentation(gt, seg)
        for key, val in res.items():
            if i == 0:
                results[key] = []
            results[key].append(val)
    print("\r " * 50, end="\r", flush=True)
    return results


def evaluate_watershed_segmentation_all_seeds(img_type, t):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    num_seeds = file["num_seeds"][()]
    file.close()
    for seed in range(1, num_seeds + 1):
        try:
            results = evaluate_watershed_segmentation(img_type, seed, t)
            while True:
                try:
                    file = h5py.File(f"results/{img_type}_results.h5", "r+")
                    break
                except BlockingIOError:
                    pass
            t_str = f"{t:.3f}".replace(".", "")
            for key, val in results.items():
                file["watershed"][f"seed_{seed}"][f"img_{t_str}"].create_dataset(key, data=val)
            file.close()
        except FileExistsError:
            print(f"Evaluation for watershed for img_type = {img_type}, seed = {seed}, t = {t} already exist")
            continue


def compute_filament_segmentations_ms(img, offsets, biases):

    probs = np.clip(img, 1e-6, 1 - 1e-6)
    vertex_costs = np.log(probs / (1-probs))
    interaction_costs = get_line_costs(offsets, vertex_costs, np.median)
    costs = np.concatenate([np.expand_dims(vertex_costs, 0), interaction_costs])

    segmentations = []
    for i, bias in enumerate(biases):
        print(f"\rmulti-separator bias {i+1} / {len(biases)}", end="", flush=True)

        shape = costs.shape[1:]
        costs = np.require(costs, requirements="F")
        flattened_costs = costs.flatten(order="F")
        gsg = GreedySeparatorGrowing3D(shape, offsets, flattened_costs)
        gsg.run()
        vertex_labels = np.array(gsg.vertex_labels())
        vertex_labels = vertex_labels.reshape(shape, order="F")
        vertex_labels = np.require(vertex_labels, requirements="C")
        segmentations.append(vertex_labels)

    print("\r " * 50, end="\r", flush=True)

    return segmentations


def compute_cell_segmentations_ms(img, offsets, biases):

    probs = np.clip(img, 1e-6, 1 - 1e-6)
    vertex_costs = np.log((1-probs) / probs)
    interaction_costs = get_line_costs(offsets, vertex_costs, np.min)

    segmentations = []
    for i, bias in enumerate(biases):
        print(f"\rmulti-separator bias {i+1} / {len(biases)}", end="", flush=True)
        gss = GreedySeparatorShrinking()
        gss.setup_grid(vertex_costs.shape, offsets.flatten(), vertex_costs.flatten() + bias,
                       interaction_costs.flatten() + bias)
        gss.run()
        seg = np.array(gss.vertex_labels()).reshape(vertex_costs.shape).astype(np.uint16)
        segmentations.append(seg)

    print("\r " * 50, end="\r", flush=True)
    return segmentations


def evaluate_segmentation(gt, seg):
    tp = np.count_nonzero(seg[gt > 0] > 0)
    tn = np.count_nonzero(seg[gt == 0] == 0)
    fp = np.count_nonzero(seg[gt == 0] > 0)
    fn = np.count_nonzero(seg[gt > 0] == 0)

    variation_of_information = VariationOfInformation(gt, seg, True)
    voi = variation_of_information.value()
    voi_fc = variation_of_information.valueFalseCut()
    voi_fj = variation_of_information.valueFalseJoin()

    num_sep = np.count_nonzero(gt == 0)
    separator_weight = (gt.size - num_sep) / num_sep
    weights = np.ones(gt.shape)
    weights[gt == 0] = separator_weight
    weighted_singleton_variation_of_information = WeightedSingletonVariationOfInformation(gt, seg, weights)
    ws_voi = weighted_singleton_variation_of_information.value()
    ws_voi_fc = weighted_singleton_variation_of_information.valueFalseCut()
    ws_voi_fj = weighted_singleton_variation_of_information.valueFalseJoin()

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "REC": tp / (tp + fn), "PRE": tp / (tp + fp) if (tp + fp) > 0 else 1,
        "VOI": voi, "VOI_FC": voi_fc, "VOI_FJ": voi_fj,
        "WS_VOI": ws_voi, "WS_VOI_FC": ws_voi_fc, "WS_VOI_FJ": ws_voi_fj,
    }


def aggregate_results(img_type, algorithm, metric):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    num_seeds = file["num_seeds"][()]
    noise_levels = file["noise_levels"]

    num_parameters = len(file[algorithm]["thresholds" if algorithm == "watershed" else "biases"])

    values = np.full((len(noise_levels), num_seeds, num_parameters), fill_value=np.nan)

    for i, t in enumerate(noise_levels):
        t_str = f"{t:.3f}".replace(".", "")
        for j, seed in enumerate(range(1, num_seeds+1)):
            try:
                values[i, j] = file[algorithm][f"seed_{seed}"][f"img_{t_str}"][metric][:]
            except KeyError:
                pass
    file.close()

    return values


def load_biases(img_type):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    biases = file["multi_separator"]["biases"][:]
    file.close()
    return biases


def load_thresholds(img_type):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    thresholds = file["watershed"]["thresholds"][:]
    file.close()
    return thresholds


def load_noise_levels(img_type):
    file = h5py.File(f"results/{img_type}_results.h5", "r")
    noise_levels = file["noise_levels"][:]
    file.close()
    return noise_levels


def plot_threshold_heatmap(t_min, t_max, value, ax=None, color="tab:blue", v_min=0, v_max=1):
    t_min_unique = np.sort(np.unique(t_min))
    t_max_unique = np.sort(np.unique(t_max))

    dt = t_min_unique[1] - t_min_unique[0]

    heatmap = np.ones((len(t_min_unique), len(t_max_unique), 3)) / 2
    value_map = np.full((len(t_min_unique), len(t_max_unique)), fill_value=np.nan)

    def cmap(v):
        v = np.clip((v - v_min) / (v_max - v_min), 0, 1)
        return v * np.array(to_rgb(color)) + (1-v)

    for i in range(len(value)):
        t1 = t_min[i]
        t2 = t_max[i]
        idx1 = np.argwhere(t_min_unique == t1)
        idx2 = np.argwhere(t_max_unique == t2)
        heatmap[idx1, idx2] = cmap(value[i])
        value_map[idx1, idx2] = value[i]

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(np.swapaxes(heatmap, 0, 1)[::-1],
              extent=[t_min_unique[0]-dt/2, t_min_unique[-1]+dt/2, t_max_unique[0]-dt/2, t_max_unique[-1]+dt/2])
    ax.imshow(np.swapaxes(value_map, 0, 1)[::-1], alpha=0,
              extent=[t_min_unique[0] - dt / 2, t_min_unique[-1] + dt / 2, t_max_unique[0] - dt / 2,
                      t_max_unique[-1] + dt / 2])


def render_best_segmentation(t, seed, img_type: str = "spline", ignore_comp_size: int = 0):
    t_str = f"{t:.3f}".replace(".", "")

    noise_levels = load_noise_levels(img_type)
    t_idx = np.argwhere(noise_levels == t)[0, 0]

    data = aggregate_results(img_type, "multi_separator", "WS_VOI")[t_idx]
    bias_idx = np.argmin(np.mean(data, axis=0))

    data = aggregate_results(img_type, "watershed", "WS_VOI")[t_idx]
    threshold_idx = np.argmin(np.mean(data, axis=0))

    file = h5py.File(f"results/{img_type}_results.h5")
    gt = file["synth_img"][f"seed_{seed}"]["gt"][:]
    img = file["synth_img"][f"seed_{seed}"][f"img_{t_str}"][:]
    seg_ms = file["multi_separator"][f"seed_{seed}"][f"img_{t_str}"]["seg"][bias_idx]
    t_min, t_max = file["watershed"]["thresholds"][threshold_idx]

    # print values
    values_to_print = ["WS_VOI", "WS_VOI_FC", "WS_VOI_FJ", "VOI", "VOI_FC", "VOI_FJ"]

    print(f"{t:.2f} & WS & {np.round(t_min, 3):.3f} & {np.round(t_max, 3):.3f} &", end=" ")
    for val in values_to_print:
        print(f"& {np.abs(np.round(file['watershed'][f'seed_{seed}'][f'img_{t_str}'][val][threshold_idx], 3)):.3f}",
              end=" ")
    print("\\\\")
    print(f"& MS & & & {np.round(file['multi_separator']['biases'][bias_idx], 2):.2f}", end=" ")
    for val in values_to_print:
        print(f"& {np.abs(np.round(file['multi_separator'][f'seed_{seed}'][f'img_{t_str}'][val][bias_idx], 3)):.3f}",
              end=" ")
    print("\\\\")
    print("\\hline")
    file.close()

    # compute watershed segmentation
    seg_ws = seeded_region_growing(img if img_type == "cell" else (1 - img), t_min, t_max,
                                   watershed_line=img_type == "cell")

    # compute voi to check that the computed segmentation is really correct
    weights = np.ones(gt.shape)
    num_sep = np.count_nonzero(gt == 0)
    weights[gt == 0] = (gt.size - num_sep) / num_sep
    ws_voi = WeightedSingletonVariationOfInformation(gt, seg_ws, weights)
    assert abs(ws_voi.value() - data[seed-1, threshold_idx]) < 1e-6

    # prepare segmentation for render
    match_labels(gt, seg_ws)
    match_labels(gt, seg_ms)

    if ignore_comp_size > 0:
        for seg in [seg_ms, seg_ws]:
            labels, counts = np.unique(seg, return_counts=True)
            for lab, count in zip(labels, counts):
                if count <= ignore_comp_size:
                    seg[seg == lab] = 0

    max_gt_label = np.max(gt)
    seg_ms[seg_ms > max_gt_label] = max_gt_label + 1
    seg_ws[seg_ws > max_gt_label] = max_gt_label + 1

    # Render
    color_list = [to_rgb(c) for c in glasbey[:max_gt_label]]
    opacity = 0.7
    color_list = [to_hex([opacity * c + (1 - opacity) for c in color]) for color in color_list]
    cmap = ListedColormap(color_list + [to_hex((0.7,)*3)])

    file_img = f"results/{img_type}_{seed}_{t_str}_img.png"
    file_gt = f"results/{img_type}_{seed}_gt.png"
    file_ms = f"results/{img_type}_{seed}_{t_str}_ms.png"
    file_ws = f"results/{img_type}_{seed}_{t_str}_ws.png"

    plot_cube(img if img_type == "cell" else 1 - img, file_name=file_img, cmap="gray")
    plot_voxels(gt, file_gt, cmap=cmap, clim=(1, max_gt_label+1))
    plot_voxels(seg_ms, file_ms, cmap=cmap, clim=(1, max_gt_label+1))
    plot_voxels(seg_ws, file_ws, cmap=cmap, clim=(1, max_gt_label+1))

    x_from, x_to = 80, 1024
    y_from, y_to = 70, 954

    for file in [file_img, file_gt, file_ms, file_ws]:
        img = plt.imread(file)
        img = img[x_from: x_to, y_from: y_to]
        plt.imsave(file, img)


def compute_ms_objective(seg, offsets, vertex_costs, interaction_costs):
    obj = np.sum(vertex_costs[seg == 0])
    for i, off in enumerate(offsets):
        slicer1 = tuple([slice(max(0, -off[d]), min(seg.shape[d], seg.shape[d] - off[d])) for d in range(seg.ndim)])
        slicer2 = tuple([slice(max(0, off[d]), min(seg.shape[d], seg.shape[d] + off[d])) for d in range(seg.ndim)])
        mask = np.logical_or(seg[slicer1] != seg[slicer2], seg[slicer1] == 0, seg[slicer2] == 0)
        obj += np.sum(interaction_costs[i][slicer1][mask])
    return obj


def compute_segmentations_ms(img_type):
    for t in load_noise_levels(img_type):
        compute_segmentation_all_seeds(img_type, t)


def evaluate_watershed(img_type):
    for t in load_noise_levels(img_type):
        evaluate_watershed_segmentation_all_seeds(img_type, t)


def evaluate_ms_segmentations(img_type):
    for t in load_noise_levels(img_type):
        for seed in range(1, 11):
            try:
                evaluate_ms_segmentation(img_type, seed, t)
            except (FileExistsError, KeyError) as e:
                print(e)


def plot_threshold_heatmaps(img_type):
    thresholds = load_thresholds(img_type)
    metrics = ["WS_VOI", "WS_VOI_FC", "WS_VOI_FJ"]

    data = {metric: aggregate_results(img_type, "watershed", metric) for metric in metrics}
    v_min = 0
    v_max = 5 if img_type == "cell" else 2

    noise_levels = load_noise_levels(img_type)
    selected_noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, ax = plt.subplots(len(metrics), len(selected_noise_levels), sharey=True, sharex=True,
                           figsize=(13, 4) if img_type == "cell" else (6, 4))

    for i, t in enumerate(selected_noise_levels):
        t_idx = np.argwhere(noise_levels == t)[0, 0]
        idx = np.argmin(np.mean(data["WS_VOI"][t_idx], axis=0))
        t_min, t_max = thresholds[idx]

        for j, metric in enumerate(metrics):
            plot_threshold_heatmap(thresholds[:, 0], thresholds[:, 1], np.median(data[metric][t_idx], axis=0),
                                   ax=ax[j, i], v_min=v_min, v_max=v_max)
            ax[j, i].scatter([t_min], [t_max], color="black", marker="+")

        ax[0, i].set_title(f"t = {t:.2f}")
        ax[-1, i].set_xlabel(r"$\theta_{start}$")

    for i in range(len(metrics)):
        ax[i, 0].set_ylabel(r"$\theta_{end}$")

    if img_type == "cell":
        fig.subplots_adjust(left=0.045, bottom=0.1, right=0.99, top=0.95, wspace=0.05, hspace=0.05)
    else:
        fig.subplots_adjust(left=0.1, bottom=0.11, right=0.99, top=0.94, wspace=0.1, hspace=0.1)
    plt.show()
    fig.savefig(f"results/{img_type}_ws_threshold_heatmaps.png", dpi=300)


def plot_against_noise_level(img_type):
    metrics = ["WS_VOI", "WS_VOI_FC", "WS_VOI_FJ", "VOI", "VOI_FC", "VOI_FJ"]
    metric_names = ["VI-WS", "FC", "FJ", "VI-NS", "FC-NS", "FJ-NS"]

    noise_levels = load_noise_levels(img_type)
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(12, 4.5))
    for i in range(1, 3):
        ax[0, i].sharey(ax[0, 0])
        ax[1, i].sharey(ax[1, 0])
    ax = ax.flatten()

    for algo, color in [("multi_separator", "tab:red"), ("watershed", "tab:blue")]:
        data = aggregate_results(img_type, algo, "WS_VOI")
        best_parameter_idx = np.argmin(np.mean(data, axis=1), axis=-1)
        for i in range(len(metrics)):
            data = aggregate_results(img_type, algo, metrics[i])
            best_data = data[range(data.shape[0]), :, best_parameter_idx]
            plot_quartile_curve(best_data, t=noise_levels[:best_data.shape[0]], ax=ax[i], color=color)
            ax[i].set_ylabel(metric_names[i])

    for i in range(3, 6):
        ax[i].set_xlabel(r"$t$")

    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.99, top=0.98, wspace=0.2, hspace=0.1)
    plt.show()
    fig.savefig(f"results/{img_type}_results.png", dpi=300)


def render_results(img_type):
    for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
        render_best_segmentation(t=t, seed=1, img_type=img_type, ignore_comp_size=0 if img_type == "cell" else 9)


def plot_bias_for_selected_noise_levels(img_type):
    biases = load_biases(img_type)
    noise_levels = load_noise_levels(img_type)
    selected_noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    metrics = ["WS_VOI_FC", "WS_VOI_FJ", "WS_VOI"]
    metric_names = ["FC", "FJ", "VI-WS"]
    colors = [GREEN, YELLOW, RED, BLUE]

    fig, ax = plt.subplots(1, len(selected_noise_levels), sharex=True, sharey=True, figsize=(12, 2))

    for metric, name, color in zip(metrics, metric_names, colors):
        data = aggregate_results(img_type, "multi_separator", metric)
        for i, t in enumerate(selected_noise_levels):
            t_idx = np.argwhere(noise_levels == t)[0, 0]
            plot_quartile_curve(data[t_idx].T, biases, ax=ax[i], color=color, name=name)
            ax[i].set_title(rf"$t = {t:.2f}$")
            ax[i].set_xlabel(r"$b$")
    ax[0].legend()

    if img_type == "filament":
        ax[0].set_ylim(-0.1, 2.6)
    else:
        ax[0].set_ylim(-0.2, 6.2)

    fig.subplots_adjust(left=0.04, bottom=0.22, right=0.99, top=0.87, wspace=0.05, hspace=0.0)
    plt.show()
    fig.savefig(f"results/{img_type}_ms_bias.png", dpi=300)


def main(img_type):

    # create the results file
    setup_results_file(img_type)

    # compute the segmentations with the multi-separator algorithm
    compute_segmentations_ms(img_type)

    # evaluate those segmentations
    evaluate_ms_segmentations(img_type)

    # compute and evaluate the segmentations with the watershed algorithm
    evaluate_watershed(img_type)

    # plot the results
    plot_against_noise_level(img_type)
    plot_bias_for_selected_noise_levels(img_type)
    plot_threshold_heatmaps(img_type)
    render_results(img_type)


if __name__ == "__main__":
    import sys
    image_type = sys.argv[1]
    if image_type not in ["filament", "cell"]:
        raise ValueError(f"invalid image type '{image_type}'. Valid image types are 'filament' or 'cell'.")

    main(image_type)
