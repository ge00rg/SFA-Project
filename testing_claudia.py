import moving_bat as mb
import numpy as np
import mdp
import matplotlib.pyplot as plt



ROOMWIDTH = 3                    #width of the room
ROOMLENGTH = 2                   #length of room


def generate_data(trajectory, sensors): 
    '''
    input: 
    trajectory= np.array of shape (n_t, 2) holding the bats position (x,y) at each timestep
    sensors= np.array (2,n) directions of n sensors_p2

    output: 
    distances=np.array of shape timesteps x sensors x distancces x wallindex
    '''
    distances=np.zeros((trajectory.shape[0], sensors.shape[1]))
    for t in range(trajectory.shape[0]): 
        distances[t,:]= mb.sonar(trajectory[t,:], sensors, width=ROOMWIDTH, length=ROOMLENGTH)[:,0]
    return distances 


tr=mb.make_trajectory()
print(tr)
print(np.round(tr, decimals=4))
mb.plot_trajectory(tr)
sen=mb.generate_sensors(n=2, direction='random')

data=generate_data(tr, sen)
print(data.shape)

flow = (mdp.nodes.PolynomialExpansionNode(1) + mdp.nodes.SFANode())
flow.train(data)
slow = flow(data)
print(slow)
print(slow.shape)
#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(tr[:,0],tr[:,1] )

#plt.subplot(1,2,2)
#plt.plot(slow[:,0],slow[:,1])

#plt.show()

#xs=np.arange(0,ROOMLENGTH, 0.001)
#ys=np.arange(0,ROOMWIDTH, 0.001)
#xx, yy=meshgrid(xs,ys)


mesh=np.zeros((ROOMWIDTH*100, ROOMLENGTH*100))

for t in range(tr.shape[0]): 
    mesh[int(100*(np.round(tr[t,0], decimals=3))),int(100*(np.round(tr[t,1],decimals=3)))]=slow[t,0]
    
plt.imshow(mesh)
plt.show() 
    




    
