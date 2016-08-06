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
        flow.insert(1, mdp.nodes.WhiteningNode(svd=svd, reduce=True))

    flow.train(series)

    return flow

def execute_sfa(flow, grid):
    '''
    returns: sfa-traeted grid (dnarray) of the same shape as grid
    '''
    return flow(grid)

def train_ica(series, in_dim=2):
    '''
    series: ndarray of share n_t x n where n_t is the numer of timesteps,
        n is the number of variables
    in_dim: int, number of dimensions admitted to ica

    return: trained ica flow
    '''
    flow = (mdp.nodes.CuBICANode(whitened=True))
    flow.train(series[:,:in_dim])

    return flow

def execute_ica(flow, grid):
    '''
    flow: flow as defined in mdp
    grid: nx2 series

    returns: ica-treated grid
    '''
    return flow(grid)

def mesh(sen, data, flow, spacing=0.1, width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, ret_dim=5, ica=False, icadim=2, draw=True, save=False, savestring=''):
    '''
    traj: ndarray as created by moving_bat.make_trajectory
    data: nxt dimensional ndarray, n number of sensors, t number of time steps,
        output of moving_bat.generate_data function
    flow: flow as defined in mdp
    spacing: float, space between two points on the used grid
    ret_dim: int, number of sfa signals returned
    draw: bool, wether to draw the resulting mesh

    retuns: the whole room treated with the trained sfa, ndarray.
    '''
    od = flow[-1].get_output_dim()
    x, y, grid = mb.generate_grid_data(sen, width, length, spacing=spacing)

    dim_temp = len(x)*len(y)
    grid_temp = np.reshape(grid, (dim_temp, sen.shape[1]))

    slow = execute_sfa(flow, grid_temp)

    if ica:
        ica_flow = (mdp.nodes.CuBICANode())
        slow = ica_flow(slow[:,:icadim])
        od = ica_flow.get_output_dim()

    slow_reshape = np.reshape(slow, (len(x), len(y), od))

    if draw:
        rng = np.max([mb.ROOMLENGTH, mb.ROOMWIDTH])
        yticks = np.arange(0, mb.ROOMWIDTH, 1)
        xticks = np.arange(0, mb.ROOMLENGTH, 1)
        for i in range(np.min([od, ret_dim])):
            plt.xticks(np.arange(0, mb.ROOMLENGTH*10,10),xticks)
            plt.yticks(np.arange(0, mb.ROOMWIDTH*10,10),yticks)
            plt.imshow(slow_reshape[:,:,i], interpolation='none', origin='lower')
            plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
            if save: 
                title=savestring+"_"+str(i)
                plt.savefig(title)
            else: 
                plt.show()
                
        
    return slow_reshape
