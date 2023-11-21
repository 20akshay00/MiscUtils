import numpy as np
import matplotlib.pyplot as plt


# construct (hyper-)cubical ranges centred at anchor w/ side-length = window
def cube(anchor, window):
    return tuple(
        slice(rmin, rmax)
        for (rmin, rmax) in list(zip(anchor - window, anchor + window))
    )


# measure of signal extent
def f(data, anchor, window):
    ranges = cube(anchor, window)
    return np.std(data[ranges])


def crop_extents(data, scale=5, max_window=200):
    """
    Returns an n-tuple of numpy array slices to crop an n-dim array

            Parameters:
                    data (np.ndarray): n-dimensional data
                    scale (int): arbitrary scaling for cropping
                    max_window (int): rough upper limit of signal extent

            Returns:
                    crop_extents (tuple): Tuple numpy array slices
    """
    anchor = np.array(
        np.unravel_index(np.argmax(data), data.shape)
    )  # replace with robust peak finding algorithm!!

    extent = np.array(
        np.argmax(
            np.array([f(data, anchor, window) for window in range(1, max_window + 1)])
        )
    )

    return cube(anchor, extent * scale)


# example usage for 2D case
if __name__ == "__main__":

    def gaussian2D(x, y, A, x0, y0, sig):
        return A * np.exp(
            -((x - x0) ** 2) / (2 * sig**2) - ((y - y0) ** 2) / (2 * sig**2)
        )

    # generate sample data
    x = np.linspace(-50, 50, 500)
    y = np.linspace(-50, 50, 500)
    X, Y = np.meshgrid(x, y)

    signal = gaussian2D(X, Y, 1, 35, 30, 2)
    noise = 0.4 * np.random.rand(len(x), len(y)) * np.random.rand(len(x), len(y))
    Z = signal + noise

    # extract crop rect -> use like arr[rng] to crop
    rng = crop_extents(Z, scale=2, max_window=50)

    # visualize
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(Z)
    ax2.imshow(Z[rng])

    plt.show()
