
### Define a raw torch loop (no netharn) for developer tests


def main():
    # Regress a 3D surface as in https://stackoverflow.com/a/53151677/887074

    # Mapping from 2D coordinates to the elevation in the 3rd dimension
    import netharn as nh
    import numpy as np
    import torch
    import ubelt as ub

    num_train = 100

    TRAIN_SURFACE = 'rosenbrock'
    if TRAIN_SURFACE == 'random':
        train_points = torch.rand(num_train, 3)
        train_XY = train_points[:, 0:2]
        train_X = train_points[:, 0:1]
        train_Y = train_points[:, 1:2]
        train_Z = train_points[:, 2:3]
    elif TRAIN_SURFACE == 'rosenbrock':
        # Train with the Rosenbrock function
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        train_points = torch.rand(num_train, 2)
        train_XY = train_points[:, 0:2]
        train_X = train_points[:, 0:1]
        train_Y = train_points[:, 1:2]

        a = 1
        b = 100
        train_Z = (a - train_X) ** 2 + b * (train_Y - train_X ** 2) ** 2 + 2

    np_train_X = train_X.data.cpu().numpy()
    np_train_Y = train_Y.data.cpu().numpy()
    np_train_Z = train_Z.data.cpu().numpy()

    test_resolution = 100
    xbasis = np.linspace(0, 1, test_resolution).astype(np.float32)
    ybasis = np.linspace(0, 1, test_resolution).astype(np.float32)
    X, Y = np.meshgrid(xbasis, ybasis)

    test_X = X.ravel()[:, None]
    test_Y = Y.ravel()[:, None]
    test_XY = np.concatenate([test_X, test_Y], axis=1)
    test_XY = torch.from_numpy(test_XY)

    import kwplot
    plt = kwplot.autoplt()
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    ax = plt.gca(projection='3d')
    ax.cla()

    # Plot the training data
    train_data_pc = ax.scatter3D(np_train_X, np_train_Y, np_train_Z, color='red')

    model = nh.layers.MultiLayerPerceptronNd(
        dim=0, in_channels=2, hidden_channels=[100] * 10, out_channels=1,
        bias=False)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    max_iters = 100

    if 0:
        iter_ = ub.ProgIter(range(max_iters), desc='iterate')
    else:
        import xdev
        iter_ = xdev.InteractiveIter(list(range(max_iters)))

    poly3d_pc = None
    for iter_idx in iter_:
        optim.zero_grad()

        pred_Z = model.forward(train_XY)
        loss = torch.nn.functional.mse_loss(pred_Z, train_Z)

        loss.backward()
        optim.step()

        test_Z = model.forward(test_XY).data.cpu().numpy()

        param_total = sum(p.sum() for p in model.parameters())
        print('param_total = {!r}'.format(param_total))

        if hasattr(iter_, 'draw'):
            num = test_X.shape[0]
            s = np.sqrt(num)
            assert s % 1 == 0
            s = int(s)
            X = test_X.reshape(s, s)
            Y = test_Y.reshape(s, s)
            Z = test_Z.reshape(s, s)
            if poly3d_pc is not None:
                # Remove previous surface
                poly3d_pc.remove()
            poly3d_pc = ax.plot_surface(X, Y, Z, color='blue')
            iter_.draw()
