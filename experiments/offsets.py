import numpy as np
from skimage.morphology import binary_dilation


def get_offsets(name: str = "default"):

    if name == "default":
        return np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0, 3, 0], [0, 3, 1], [0, 2, 2], [0, 1, 3], [0, 0, 3], [0, -1, 3], [0, -2, 2], [0, -3, 1],
            [0, 10, 0], [0, 9, 3], [0, 7, 7], [0, 3, 9], [0, 0, 10], [0, -3, 9], [0, -7, 7], [0, -9, 3],
            [2, 0, 0]
        ])


def sphere_offsets(radius: tuple):

    mesh = np.stack(np.meshgrid(*[np.arange(2*r + 1) for r in radius]))
    mesh -= np.reshape(radius, (-1,) + (1,) * len(radius))
    inside_sphere = np.sum([(mesh[i] / radius[i])**2 for i in range(len(radius))], axis=0) < 1
    sphere = binary_dilation(inside_sphere)
    sphere[inside_sphere] = False

    offsets = []
    for offset in np.argwhere(sphere) - radius:
        if tuple(-offset) not in offsets:
            offsets.append(tuple(offset))

    return np.array(offsets)


def sparse_sphere_offsets(radius: tuple, connectivity: int):

    assert connectivity in [6, 18, 26, 50]

    offsets = np.zeros((connectivity//2, 3))
    offsets[:3] = np.eye(3)

    if connectivity >= 18:
        offsets[3:9] = [
            [0, 1, 1],
            [0, 1, -1],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, 1],
            [1, 0, -1]
        ]
    if connectivity >= 26:
        offsets[9:13] = [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1]
        ]
    if connectivity >= 50:
        offsets[13:] = [
            [8, 3, 3],
            [8, 3, -3],
            [8, -3, 3],
            [8, -3, -3],
            [3, 8, 3],
            [3, 8, -3],
            [3, -8, 3],
            [3, -8, -3],
            [3, 3, 8],
            [3, 3, -8],
            [3, -3, 8],
            [3, -3, -8]
        ]
    offsets /= np.expand_dims(np.linalg.norm(offsets, axis=1), axis=-1)
    offsets *= radius
    return np.rint(offsets).astype(int)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    r = 7
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    x, y, z = np.indices((2*r+1, 2*r+1, 2*r+1))
    x -= r
    y -= r
    z -= r
    voxels = (x**2 + y ** 2 + z**2) <= r**2
    colors = np.full(voxels.shape, "blue")

    offsets = sparse_sphere_offsets((r, r, r), 50)

    all_offsets = np.concatenate([offsets, -offsets])

    for off in all_offsets:
        msk = (x == off[0]) * (y == off[1]) * (z == off[2])
        voxels[msk] = True
        colors[msk] = "red"

    ax.voxels(voxels, facecolors=colors, edgecolors="k")
    ax.set_aspect('equal')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    offsets = np.concatenate([np.eye(3, dtype=int), sparse_sphere_offsets((5,) * 3, connectivity=26)])
    all_offsets = np.concatenate([offsets, -offsets])
    ax.scatter(*all_offsets.T, alpha=1)
    for off in all_offsets:
        ax.plot([0, off[0]], [0, off[1]], [0, off[2]], color="black")
    ax.set_aspect("equal")

    plt.show()
