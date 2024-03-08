import os
import h5py
import numpy as np


def generate_spline_data(
    image_size: int = 64,
    seed: int = 1,
    number_of_splines: int = 10,
    max_num_tries_per_spline: int = 1000,
    min_spline_separation: float = 0.2,
    spline_radius: float = 0.02,
    max_p_structure: float = 0.98,
    mu_background: float = 0.3,
    sigma_background: float = 0.06,
    mu_structure: float = 0.7,
    sigma_structure: float = 0.15,
    mixture: bool = True,
    d2_range: float = 55.0,
    d2_dir_bias: float = 1,
    d2_dir_bias_weight: float = 0.5,
    spline_segment_length_base: float = 0.15,
    spline_segment_length_variance: float = 0.15,
    max_num_threads: int = 1
):
    theta_1 = -np.log(1 - max_p_structure) + np.log(max_p_structure)
    theta_2 = - theta_1 / spline_radius

    path = "data/spline_variables.config"
    if os.path.isfile(path):
        os.remove(path)

    with open(path, 'x') as f:
        f.write(
            f'''"show_visualization" : false
"max_num_threads": {max_num_threads}
"image_size" : {image_size}
"write_output_images" : true
"write_as_hdf5" : true
"append_date_suffix_to_filename" : false
"seed" : {seed}
"number_of_splines" : {number_of_splines}
"max_num_tries_per_spline" : {max_num_tries_per_spline}
"min_spline_seperation" : {min_spline_separation}
"theta_1" : {theta_1}
"theta_2" : {theta_2}
"mu_background" : {mu_background}
"sigma_background" : {sigma_background}
"mu_structure" : {mu_structure}
"sigma_structure" : {sigma_structure}
"mixture": {'true' if mixture else 'false'}
"d2_range" : {d2_range}
"d2_dir_bias" : {d2_dir_bias}
"d2_dir_bias_weight" : {d2_dir_bias_weight}
"spline_segment_length_base" : {spline_segment_length_base}
"spline_segment_length_variance" : {spline_segment_length_variance}
''')
    out = os.system("./SynImg_Splines")
    if out != 0:
        raise RuntimeError(f"os call failed: {out}")

    file_name = f"syn_splines_{image_size}_data.h5"
    file = h5py.File(file_name, "r")
    image = file["grayscale_image_group"]["grayscale_dataset"][:]
    labels = file["groundtruth_image_group"]["groundtruth_dataset"][:].astype(np.uint16)
    file.close()

    np.random.seed(seed)
    image = (image + np.random.random(image.shape)) / 256

    return image, labels


def get_spline_kwargs(t, size=64):

    ms = [0.7, 0.62]
    mb = [0.3, 0.38]
    ss = [0.05, 0.1]
    sb = [0.05, 0.1]

    assert mb[0] < mb[1] < ms[1] < ms[0]
    assert ss[0] < ss[1]
    assert sb[0] < sb[1]

    kwargs = {
        "number_of_splines": 15,
        "image_size": size,
        "mu_structure": (1 - t) * ms[0] + t * ms[1],
        "sigma_structure": (1 - t) * ss[0] + t * ss[1],
        "mu_background": (1 - t) * mb[0] + t * mb[1],
        "sigma_background": (1 - t) * sb[0] + t * sb[1],
        "max_p_structure": 0.9,
        "spline_radius": 0.75 / size,
        "min_spline_separation": 10 / size,
        "mixture": False,
        "d2_range": 55
    }
    return kwargs


def plot_distance_distribution():
    from matplotlib import pyplot as plt
    radius = 0.1

    d = np.linspace(0, 1, 1000)
    for t in np.linspace(0, 1, 100):
        p_structure = 1 - t**3/2

        p_structure = min(p_structure, 1-1e-12)

        theta_1 = -np.log(1 - p_structure) + np.log(p_structure)
        theta_2 = - theta_1 / radius

        p = 1 / (1 + np.exp(-theta_1 - theta_2 * d))
        plt.plot(d, p)
    plt.show()


if __name__ == "__main__":
    plot_distance_distribution()
    img, gt = generate_spline_data(100, mixture=False, max_num_threads=1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img[50], cmap="gray")
    ax[1].imshow(gt[50], cmap="tab20")
    plt.show()




