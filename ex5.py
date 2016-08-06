import numpy as np
import matplotlib.pyplot as plt
import moving_bat as mb
import sfa_tasks as sfa
import mdp



def plot_sensors(sen): 
    for i in range(sen.shape[1]):
        plt.plot([0, sen[0,i]], [0,sen[1,i]])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()




#Now vary the degree of the polynomial expansion (try e.g. 3, 5 and 7). Plot the position-
#dependence of the different output signals of SFA. How do the learned features change as the
#expansion degree increases? What kind of representation does SFA learn for high degrees of
#the expansion? Change the proportions of the room from quadratic to strongly elongated.
#What is the effect on the learned spatial representation? Discuss potential similarities with
#and differences to spatial representations found in rodents (and bats, as well, by the way





traj = mb.make_trajectory()
sen = np.array([[1,0,-1,0], [0,1,0,-1]])
print(sen)
plot_sensors(sen)
data = mb.generate_data(traj, sen)
flow = sfa.train_sfa(data, poly_exp=7, whiten=True, svd=True)
sfa.mesh(sen, data, flow)

























