import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.utils import shuffle as util_shuffle


# examples from :
# https://github.com/LukasRinder/normalizing-flows/blob/master/data/toy_data.py


def generate_2d_data(data, rng=None, batch_size=1000):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.max(data)


    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data, np.max(data)

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), np.max(X)

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2], dtype="float32")
        return data, np.max(data)

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return data, np.max(data)

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return np.array(x, dtype='float32'), np.max(x)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return np.array(data, dtype="float32"), np.max(data)

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
        return data, np.max(data)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        return data, np.max(data)
    elif data == "tum":
        mesh = np.zeros((400,400), dtype="float32")
        mesh[20:140, 100:130] = 1
        mesh[70:100, 120:300] = 1
        mesh[140:170, 100:300] = 1
        mesh[170:220, 250:300] = 1
        mesh[220:250, 100:300] = 1
        mesh[250:370, 100:130] = 1
        mesh[280:310, 120:300] = 1
        mesh[340:370, 120:300] = 1

        index = np.argwhere(mesh == 1)

        coordinates = index - 200
        coordinates[:,1] *= -1
        coordinates = coordinates / 50

        index_2 = np.random.randint(len(coordinates), size=batch_size)

        dataset = np.array(coordinates[index_2,:], dtype="float32")
        return dataset, np.max(dataset)
    else:
        data = generate_2d_data("8gaussians", rng, batch_size)
        return data, np.max(data)



def plot_kepler(
        x_exact: torch.Tensor, y_exact: torch.Tensor, 
        x_obs: torch.Tensor, y_obs: torch.Tensor,
        x_pred = None, y_pred = None,
    ) -> None:

    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    
    ax[0].plot(x_exact.numpy(), y_exact.numpy(), 'k-', linewidth=2, label='True')
    ax[0].scatter(0, 0, c='orange', s=200, marker='*', label='Focus')
    ax[0].scatter(x_obs.numpy(), y_obs.numpy(), c='blue', s=30, alpha=0.7, label='Data')
    
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Orbit')
    ax[0].legend(loc='upper right', fontsize=7)
    ax[0].grid(True, alpha=0.3)

    r_exact = torch.sqrt(x_exact**2 + y_exact**2)
    phi_exact = torch.atan2(y_exact, x_exact)
    
    r_obs = torch.sqrt(x_obs**2 + y_obs**2)
    phi_obs = torch.atan2(y_obs, x_obs)
    
    sort_idx_exact = torch.argsort(phi_exact)
    phi_exact_sorted = phi_exact[sort_idx_exact]
    r_exact_sorted = r_exact[sort_idx_exact]
    
    sort_idx_obs = torch.argsort(phi_obs)
    phi_obs_sorted = phi_obs[sort_idx_obs]
    r_obs_sorted = r_obs[sort_idx_obs]
    
    ax[1].plot(phi_exact_sorted.numpy(), r_exact_sorted.numpy(), 'k-', linewidth=2, label='True')
    ax[1].scatter(phi_obs_sorted.numpy(), r_obs_sorted.numpy(), c='blue', s=30, alpha=0.7, label='Data')

    ax[1].set_xlabel('φ [rad]')
    ax[1].set_title('r(φ) - Prediction/Test vs Observed')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)


    if x_pred is not None and y_pred is not None:
        ax[0].plot(x_pred.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='Predicted')
        ax[0].set_title('Prediction vs True')

    plt.tight_layout()
    