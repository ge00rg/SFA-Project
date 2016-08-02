import numpy as np
import matplotlib.pyplot as plt
import moving_bat as mb
import mdp

def train_sfa(series, poly_exp=1, out_dim=1, whiten=False, svd=False):
    '''
    series: ndarray of shape n_t x n where n_t is the number of timesteps,
        n is the number of variables
    poly_exp: int, degree of polynomial expansion
    out_sim: int, number of slowest signals

    returns: out_dim x n_t shaped ndarray, containing the first slowest signals
        extracted from series using the SFA algorithm
    '''
    flow = (mdp.nodes.PolynomialExpansionNode(poly_exp) +
            mdp.nodes.SFANode())

    if whiten:
        flow.insert(0,mdp.nodes.WhiteningNode(svd=svd, reduce=True))
    
    flow.train(series)

    return flow

def execute_sfa(flow, grid):
    '''
    returns: sfa-traeted grid (dnarray) of the same shape as grid
    '''
    return flow(grid)

def mesh(sen, data, flow, spacing=0.1, draw=True):
    '''
    traj: ndarray as created by moving_bat.make_trajectory
    data: nxt dimensional ndarray, n number of sensors, t number of time steps,
        output of moving_bat.generate_data function
    flow: flow as defined in mdp
    spacing: float, space between two points on the used grid
    draw: bool, wether to draw the resulting mesh

    retuns: the whole room treated with the trained sfa, ndarray.
    '''
    od = flow[-1].get_output_dim()
    print(od)
    x, y, grid = mb.generate_grid_data(sen, spacing=spacing)

    dim_temp = len(x)*len(y)
    grid_temp = np.reshape(grid, (dim_temp, od))

    slow = execute_sfa(flow, grid_temp)

    slow_reshape = np.reshape(slow, (len(x), len(y), od))

    if draw:
        rng = np.max([mb.ROOMLENGTH, mb.ROOMWIDTH])
        for i in range(od):
            plt.imshow(slow_reshape[:,:,i], vmin=-rng, vmax=rng, cmap='plasma')
            plt.colorbar()
            plt.show()

        return slow_reshape
