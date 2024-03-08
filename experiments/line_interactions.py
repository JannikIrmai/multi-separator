import numpy as np


def interaction2line(e):
    path = []
    max_d = np.argmax(np.abs(e))
    for i in range(0, abs(e[max_d]) + 1):
        coord = ()
        for d in range(len(e)):
            coord += (int(np.round((i * e[d]) / abs(e[max_d]))),)
        path.append(coord)
    return path


def get_line_array(offset: np.ndarray, img: np.ndarray) -> np.ndarray:
    path = interaction2line(offset)
    path_img = np.zeros((len(path),) + img.shape)
    for j, diff in enumerate(path):
        slicer_a = tuple(slice(max(0, -diff[i]), min(img.shape[i], img.shape[i] - diff[i]))
                         for i in range(len(offset)))
        slicer_b = tuple(slice(max(0, diff[i]), min(img.shape[i], img.shape[i] + diff[i]))
                         for i in range(len(offset)))
        path_img[(j,) + slicer_a] = img[slicer_b]
    return path_img


def get_line_costs(offsets: np.ndarray, img: np.ndarray, method=np.max) -> np.ndarray:
    line_costs = np.zeros((offsets.shape[0],) + img.shape, dtype=img.dtype)
    for i, offset in enumerate(offsets):
        path_img = get_line_array(offset, img)
        line_costs[i] = method(path_img, axis=0)
    return line_costs


def main():
    import matplotlib.pyplot as plt

    shape = (8, 6)
    e = (5, 3)
    dx, dy = (1, 1)

    line = interaction2line(e)
    pixels = [(x+dx, y+dy) for x, y in line]

    fig, ax = plt.subplots()

    img = np.ones(shape)
    for x, y in pixels:
        img[x, y] = 0.5

    ax.imshow(img.T, cmap="gray", vmax=1, vmin=0)

    for i in range(1, shape[1]):
        ax.plot([-0.5, shape[0] - 0.5], [i - 0.5, i - 0.5], color="black")
    for i in range(1, shape[0]):
        ax.plot([i - 0.5, i - 0.5], [-0.5, shape[1] - 0.5], color="black")
    ax.plot([dx, dx + e[0]], [dy, dy+e[1]], color="red", marker="o")

    ax.set_xlim(-0.5, shape[0] - 0.5)
    ax.set_ylim(-0.5, shape[1] - 0.5)
    plt.show()


if __name__ == "__main__":
    main()
