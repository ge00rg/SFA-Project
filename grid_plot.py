import moving_bat as mb
import numpy as np
import mdp
import matplotlib.pyplot as plt
import geometry as geo



ROOMWIDTH = 3                    #width of the room
ROOMLENGTH = 2                   #length of room


def generate_data(trajectory, sensors): 
    '''
    input: 
    trajectory= np.array of shape (n_t, 2) holding the bats position (x,y) at each timestep
    sensors= np.array (2,n) directions of n sensors_p2

    output: 
    distances=np.array of shape timesteps x sensors 
    '''
    distances=np.zeros((trajectory.shape[0], sensors.shape[1]))
    for t in range(trajectory.shape[0]): 
        distances[t,:]= mb.sonar(trajectory[t,:], sensors, width=ROOMWIDTH, length=ROOMLENGTH)[:,0]
    return distances 
    
def generate_grid_data(sensors, width=ROOMWIDTH, length=ROOMLENGTH, spacing=0.1): 
    '''
    input: 
    trajectory= np.array of shape (n_t, 2) holding the bats position (x,y) at each timestep
    sensors= np.array (2,n) directions of n sensors_p2

    output: 
    distances=np.array of shape roomwidth/spacing x roomlength/spacing x sensors 
    '''
    x=np.arange(0, width, spacing)
    y=np.arange(0, length, spacing)
     
    distances=np.zeros((len(x), len(y), sensors.shape[1]))
    for i in range(len(x)): 
        for j in range(len(y)): 
            distances[i, j,:]= mb.sonar((x[i],y[j]), sensors, width=ROOMWIDTH, length=ROOMLENGTH)[:,0]
    return x,y,distances 



#generate a trajectory for training the SFA-Node
tr=mb.make_trajectory()

#generate sensors
sen=mb.generate_sensors(n=3, direction='random')

#generate sensory data 
data=generate_data(tr, sen)

#generate SFA-Node and train with the above generated data
flow = (mdp.nodes.PolynomialExpansionNode(1) + mdp.nodes.SFANode(output_dim=3))
flow.train(data)

#generate sonar data for every gridpoint in the arena
x,y,grid=generate_grid_data(sen)

#reshape the grid data to the same dimension as the training time series
new_dim=len(x)*len(y)
grid_reshape=np.reshape(grid, (new_dim, sen.shape[1]))

#perform SFA on grid-data
slow = flow(grid_reshape)

#reshape the result of SFA, 3 corresponds to number of sensors used for data generateion
slow_reshape=np.reshape(slow, (len(x), len(y), 3))

#plot stuff, (again:  3 corresponds to number of sensors used for data generateion)
for i in range(3): 
    plt.imshow(slow_reshape[:,:,i])
    plt.show() 
    
	




    
