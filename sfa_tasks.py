import numpy as np
import matplotlib.pyplot as plt
import moving_bat as mb
import mdp

def train_sfa(series, poly_exp=1, out_dim=1):
    '''
    series: ndarray of shape n_t x n where n_t is the number of timesteps,
        n is the number of variables
    poly_exp: int, degree of polynomial expansion
    out_sim: int, number of slowest signals

    returns: out_dim x n_t shaped ndarray, containing the first slowest signals
        extracted from series using the SFA algorithm
    '''
    flow = (mdp.nodes.PolynomialExpansionNode(poly_exp) +
            mdp.nodes.SFANode(output_dim=out_dim))
    
    flow.train(series)

    return flow

def execute_sfa(flow, grid):
    '''
    returns: sfa-traeted grid (dnarray) of the same shape as grid
    '''
    return flow(grid)

def mesh(traj, n=3, poly_exp=1, out_dim=3, direction='random', spacing=0.1, draw=True):
    '''
    traj: ndarray as created by moving_bat.make_trajectory
    n: int, number of sensors
    out_dim: int, number of sfa-signals
    direction: 'random' or 'orthogonal', refers to the direction in which the sensors point
    spacing: float, space between two points on the used grid
    draw: bool, wether to draw the resulting mesh

    retuns: the whole room treated with the trained sfa, ndarray.
    '''
    sen = mb.generate_sensors(n=n, direction=direction)
    data = mb.generate_data(traj, sen)
    flow = train_sfa(data, poly_exp=poly_exp, out_dim=out_dim)
    
    x, y, grid = mb.generate_grid_data(sen, spacing=spacing)

    dim_temp = len(x)*len(y)
    grid_temp = np.reshape(grid, (dim_temp, sen.shape[1]))

    slow = execute_sfa(flow, grid_temp)

    slow_reshape = np.reshape(slow, (len(x), len(y), sen.shape[1]))

    if draw:
        rng = np.max([mb.ROOMLENGTH, mb.ROOMWIDTH])
        for i in range(sen.shape[1]):
            plt.imshow(slow_reshape[:,:,i], vmin=-rng, vmax=rng, cmap='plasma')
            plt.colorbar()
            plt.show()

        return slow_reshape
