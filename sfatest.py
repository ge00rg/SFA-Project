import mdp
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,2*np.pi,1000, dtype='float64')

x1 = np.sin(t) + np.cos(11*t)**2
x2 = np.cos(11*t)

flow = (mdp.nodes.EtaComputerNode() +
        mdp.nodes.PolynomialExpansionNode(2) +
        mdp.nodes.SFANode(output_dim=1) +
        mdp.nodes.EtaComputerNode() )

series = np.zeros((1000,2), dtype='float64')
series[:,0] = x1
series[:,1] = x2

flow.train(series)
slow = flow(series)


plt.plot(t,slow)
plt.show()
