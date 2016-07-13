### ### Here be imports ### ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

### ### Here be the realm of all variables global ### ###
ROOMWIDTH = 3                    #width of the room
ROOMLENGTH = 2                   #length of room

T = 1500                         #total number of timesteps
DIRINTERVAL = 20                 #every DIRINTERVAL timesteps, a new direction vector is chosen

MAXSPEED = 0.05                  #maximum speed of the bat      
TRDIST=0.05                      #distance at which the mirroring is triggered. Corrently not in use        

### ### Here liveth the heart of the program ### ###
def get_distances(x, y, width=ROOMWIDTH, length=ROOMLENGTH):
    '''
    Assuming room extends into the first quadrant from (0,0), takes x and y coordinates and
    returns the distances to all walls.
    x: float, x-coordinate.
    y: float, y-coordinate.

    Returns: tuple (distance_west, distance_east, distance_sout, distance_north)
    '''
    return x, width-x, y, length-y

def make_trajectory(width=ROOMWIDTH, length=ROOMLENGTH, maxspeed=MAXSPEED, n_t=T, ival=DIRINTERVAL,
interpolation='linear', trigger_distance=TRDIST, init=None):
    '''
    width: int, width of the room.
    length: int, length of the room.
    maxspeed: float, maximum distance to move in one timestep.
    n_t: int, number of timesteps.
    ival: int, each ival timesteps, a new direction vector is chosen and in between two
        such vectors, we interpoate.
    interpolation: so far, only 'linear' option exists.
    trigger_distance: float. Not used yet, can be used to set the distance to the wall
        at which the mirroring kicks in if it should be different than maxspeed.
    init: array-like, shape (1,2). Starting point of trajectory. If None, it is chosen
        randomly.

    Returns: ndarray, shape (n_t,2) 2-dimensional timeseries that describes the trajectory.
    '''
    assert(n_t%ival == 0)
    #the number of timesteps should be evenly divisible by the interval
    
    if not init:
        init = [np.random.uniform(maxspeed, width-maxspeed), np.random.uniform(maxspeed, length-maxspeed)]
    else:
        print('Put assert statement here if this ever gets used!')
    #make random starting point, unless one is specified

    phi_arr = np.random.uniform(0, 2*np.pi, int(n_t/ival))
    r_arr = np.random.uniform(0, maxspeed, int(n_t/ival))
    #generate n_t/ival vectors in polar coordinates (easier to maintain normalization that way)

    x_arr = r_arr*np.cos(phi_arr)
    y_arr = r_arr*np.sin(phi_arr)
    #convert to cartesian coordinates

    vs = np.empty((n_t, 2))
    #will hold interpolated vectors for each timestep
    
    if interpolation == 'linear':
        for i in range(int(n_t/ival-1)):
            x_temp = np.linspace(x_arr[i], x_arr[i+1], ival)
            y_temp = np.linspace(y_arr[i], y_arr[i+1], ival)
            #interpolation
            
            vs[i*ival : ival + i*ival ,0] = x_temp
            vs[i*ival : ival + i*ival, 1] = y_temp
            #saving to vs
    
    traj = np.empty((n_t,2))
    traj[0] = init
    #initialize trajectory

    for t in range(1, n_t):
        traj[t] = traj[t-1] + vs[t-1]
        w, e, s, n = get_distances(traj[t][0], traj[t][1])
        if w < maxspeed or e < maxspeed:
            vs[t-1:, 0] *= -1
            traj[t] = traj[t-1] + vs[t-1]
        if n < maxspeed or s < maxspeed:
            vs[t-1:, 1] *= -1
            traj[t] = traj[t-1] + vs[t-1]
    #adding elements of vs to trajectory at every timestep, morroring everything in the future
    #if conditions are met
    return traj

def plot_trajectory():
    a = make_trajectory()

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            ROOMWIDTH,
            ROOMLENGTH,
            fill=False      # remove background
        )
    )
    ax.plot(a[:,0],a[:,1])
    ax.scatter(a[0,0], a[0,1])
    ax.set_xlim([-0.2,ROOMWIDTH+0.2])
    ax.set_ylim([-0.2,ROOMLENGTH+0.2])
    plt.show()
