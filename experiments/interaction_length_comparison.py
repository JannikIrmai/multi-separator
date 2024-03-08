import numpy as np
from offsets import sphere_offsets, sparse_sphere_offsets
import h5py
from generate_cell_data import get_cell_kwargs, generate_cell_data
from generate_spline_data import get_spline_kwargs, generate_spline_data
from experiments import (compute_cell_segmentations_ms, compute_filament_segmentations_ms, evaluate_segmentation,
                         plot_quartile_curve)
import matplotlib.pyplot as plt


def setup_results_file(img_type="filament"):

    num_seeds = 10

    radii = list(range(2, 11))

    if img_type == "filament":
        biases = np.round(np.linspace(-0.05, 0.2, 21), 3)
    elif img_type == "cell":
        biases = np.round(np.linspace(-0.35, 0.15, 21), 3)
    else:
        raise ValueError(f"Invalid img_type '{img_type}'")

    noise_level = 0.5

    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "w-")
    file.create_dataset("num_seeds", data=num_seeds)
    file.create_dataset("noise_level", data=noise_level)
    file.create_dataset("radii", data=radii)

    synth_img_group = file.create_group("synth_img")

    ms_group = file.create_group("multi_separator")
    ms_group.create_dataset("biases", data=biases)

    get_kwargs = get_spline_kwargs if img_type == "filament" else get_cell_kwargs
    generate_data = generate_spline_data if img_type == "filament" else generate_cell_data

    for seed in range(1, num_seeds + 1):
        print("seed:", seed)
        seed_group = synth_img_group.create_group(f"seed_{seed}")
        seed_group_ms = ms_group.create_group(f"seed_{seed}")

        kwargs = get_kwargs(noise_level)
        img, gt = generate_data(seed=seed, **kwargs)
        seed_group.create_dataset("gt", data=gt)
        seed_group.create_dataset(f"img", data=img)
        for r in radii:
            seed_group_ms.create_group(f"radius_{r}")

    file.close()


def compute_segmentations(img_type, seed, radius):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r")
    ms_group = file["multi_separator"]
    seed_group = file["synth_img"][f"seed_{seed}"]
    if f"seg" in file["multi_separator"][f"seed_{seed}"][f"radius_{radius}"]:
        raise FileExistsError(f"Segmentations for img_type = {img_type}, seed = {seed}, radius={radius} already exist.")

    img = seed_group[f"img"][:]

    if img_type == "filament":
        offsets = np.concatenate([np.eye(3, dtype=int), sphere_offsets((radius,) * 3)])
    else:
        offsets = np.concatenate([np.eye(3, dtype=int), sparse_sphere_offsets((radius,) * 3, 26)])

    biases = ms_group["biases"][:]
    file.close()

    print(f"Running ms for img_type = {img_type}, seed = {seed}, radius = {radius}")
    ms_algo = compute_filament_segmentations_ms if img_type == "filament" else compute_cell_segmentations_ms
    segmentations = ms_algo(img, offsets, biases)

    while True:
        try:
            file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r+")
            group = file["multi_separator"][f"seed_{seed}"][f"radius_{radius}"]
            group.create_dataset(f"seg", data=segmentations)
            file.close()
            break
        except BlockingIOError:
            print(f"\rCannot save segmentation, file blocked!", end="", flush=True)
            pass


def compute_segmentation_all_seeds(img_type, radius):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r")
    num_seeds = file["num_seeds"][()]
    file.close()
    for seed in range(1, num_seeds + 1):
        try:
            compute_segmentations(img_type, seed, radius)
        except FileExistsError:
            print(f"segmentations for img_type = {img_type}, seed = {seed}, radius = {radius} already exist")
            continue


def evaluate_ms_segmentation(img_type, seed, radius):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r+")

    if len(file["multi_separator"][f"seed_{seed}"][f"radius_{radius}"]) > 1:
        raise FileExistsError(f"MS results for img_type={img_type}, seed={seed}, radius={radius} already exist")

    print(f"Evaluating MS for img_type = {img_type}, seed = {seed}, radius = {radius}")

    gt = file["synth_img"][f"seed_{seed}"]["gt"][:]
    ms_group = file["multi_separator"][f"seed_{seed}"][f"radius_{radius}"]
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


def load_radii(img_type):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r")
    radii = file["radii"][:] if "radii" in file else np.arange(2, 11)
    file.close()
    return radii


def load_biases(img_type):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r")
    biases = file["multi_separator"]["biases"][:]
    file.close()
    return biases


def aggregate_results(img_type, metric):
    file = h5py.File(f"results/{img_type}_offset_range_results.h5", "r")
    num_seeds = file["num_seeds"][()]
    radii = file["radii"][:] if "radii" in file else np.arange(2, 11)

    num_biases = len(file["multi_separator"]["biases"])

    values = np.full((len(radii), num_seeds, num_biases), fill_value=np.nan)

    for i, r in enumerate(radii):
        for j, seed in enumerate(range(1, num_seeds+1)):
            try:
                 values[i, j] = file["multi_separator"][f"seed_{seed}"][f"radius_{r}"][metric][:]
            except KeyError:
                pass
    file.close()

    return values


def compute_all_segmentations(img_type):
    radii = load_radii(img_type)
    for r in radii:
        compute_segmentation_all_seeds(img_type, r)


def evaluate_segmentations(img_type):
    for r in load_radii(img_type):
        for seed in range(1, 11):
            try:
                evaluate_ms_segmentation(img_type, seed, r)
            except (FileExistsError, KeyError) as e:
                print(e)


def plot_by_radius(img_type):
    data = aggregate_results(img_type, "WS_VOI")
    radii = load_radii(img_type)
    best_bias_idx = np.argmin(np.mean(data, axis=1), axis=-1)

    metrics = ["WS_VOI", "WS_VOI_FC", "WS_VOI_FJ"]
    metric_names = ["VI-WS", "FC", "FJ"]

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(12, 2.5))
    for i in range(1, 3):
        ax[i].sharey(ax[0])

    for i in range(len(metrics)):
        data = aggregate_results(img_type, metrics[i])
        best_data = data[range(data.shape[0]), :, best_bias_idx]
        plot_quartile_curve(best_data, t=radii, ax=ax[i], color="tab:red")
        ax[i].set_ylabel(metric_names[i])

    for i in range(3):
        ax[i].set_xlabel(r"$r$")

    fig.subplots_adjust(left=0.06, bottom=0.2, right=0.99, top=0.96, wspace=0.2, hspace=0.1)
    plt.show()
    fig.savefig(f"results/{img_type}_interaction_length_results.png", dpi=300)


def main(img_type):

    # create the results file
    setup_results_file(img_type)

    # compute the segmentations with the multi-separator algorithm
    compute_all_segmentations(img_type)

    # evaluate those segmentations
    evaluate_segmentations(img_type)

    # plot the results
    plot_by_radius(img_type)


if __name__ == "__main__":
    import sys
    image_type = sys.argv[1]
    if image_type not in ["filament", "cell"]:
        raise ValueError(f"invalid image type '{image_type}'. Valid image types are 'filament' or 'cell'.")

    main(image_type)
