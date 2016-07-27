import numpy as np
import matplotlib.pyplot as plt
import moving_bat as mb
import mdp

def do_sfa(series, poly_exp=7, out_dim=1):
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

    return flow(series)

