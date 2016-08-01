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
    flow = (mdp.nodes.EtaComputerNode() +
            mdp.nodes.PolynomialExpansionNode(poly_exp) +
            mdp.nodes.SFANode(output_dim=out_dim) +
            mdp.nodes.EtaComputerNode() )
    
    flow.train(series)

    return flow

def execute_sfa(flow, grid):
    return flow(grid)

#def mesh(traj, slow, width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, reso=100):
#    assert(traj.shape[0] == slow.shape[0])
#    mesh = np.zeros((width*reso, length*reso))
#
#    for t in range(traj.shape[0]):
#            mesh[int(reso*(np.round(traj[t,0], decimals=3))), int(reso*(np.round(traj[t,1], decimals=3)))] = slow[t,0]
#    
#    return mesh

def mesh(traj, n=3, poly_exp=1, out_dim=3, direction='random', width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, spacing=0.1, draw=True):
    sen = mb.generate_sensors(n=n, direction=direction)
    data = mb.generate_data(traj, sen)
    flow = train_sfa(data, poly_exp=poly_exp, out_dim=out_dim)
    
    x, y, grid = mb.generate_grid_data(sen, spacing=spacing)

    dim_temp = len(x)*len(y)
    grid_temp = np.reshape(grid, (dim_temp, sen.shape[1]))

    slow = execute_sfa(flow, grid_temp)

    slow_reshape = np.reshape(slow, (len(x), len(y), sen.shape[1]))

    if draw:
        for i in range(sen.shape[1]):
            plt.imshow(slow_reshape[:,:,i])
            plt.show()

        return slow_reshape
