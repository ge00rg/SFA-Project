### ### Here be imports ### ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import geometry as geo

### ### Here be the realm of all variables global ### ###
ROOMWIDTH = 3                    #width of the room
ROOMLENGTH = 2                   #length of room

T = 1500                         #total number of timesteps
DIRINTERVAL = 20                 #every DIRINTERVAL timesteps, a new direction vector is chosen

MAXSPEED = 0.05                  #maximum speed of the bat      
MINSPEED = 0.005                 #minimum speed of the bat
TRDIST = 0.05                      #distance at which the mirroring is triggered. Corrently not in use

WALLSDICT = {0:'south', 1:'east', 2:'north', 3:'west', 'south':0, 'east':1, 'north':2, 'west':3}

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

def make_trajectory(width=ROOMWIDTH, length=ROOMLENGTH, minspeed=MINSPEED, maxspeed=MAXSPEED, n_t=T, ival=DIRINTERVAL,
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
    r_arr = np.random.uniform(minspeed, maxspeed, int(n_t/ival))
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

def sensory_data(traj, sensors):
    '''
    traj: trajectory as returned by make_trajectory
    n: int, number of sensors
    direction: 'random' or 'orthogonal'

    returns: ndarray(txn) containing the sensory data obtained from traj sensory data 
    '''
    sen_data = np.zeros((traj.shape[0], sensors.shape[1]))
    for t in range(traj.shape[0]):
        sen_data[t] = sonar(traj[t], sensors)[:,0]

    return sen_data

def plot_trajectory(traj):
    '''
    Plots a random trajectory.
    '''

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
    ax.plot(traj[:,0],traj[:,1])
    ax.scatter(traj[0, 0], traj[0, 1])
    ax.scatter(traj[-1,0], traj[-1, 1], color='r')
    ax.set_xlim([-0.2,ROOMWIDTH+0.2])
    ax.set_ylim([-0.2,ROOMLENGTH+0.2])
    plt.show()

def test_sonar(pos):
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
    plt.scatter(pos[0], pos[1])
    
    sensors = generate_sensors(3)
    son = sonar(pos, sensors)
    for i in range(sensors.shape[1]):
        plt.plot([pos[0], pos[0] + sensors[0,i]*son[i,0]], [pos[1], pos[1] + sensors[1,i]*son[i,0]])


    plt.show()


def generate_sensors(n=2, direction='random'):
    '''
    n: int, number of desired sensors
    direction: 'random' or 'orthogonal', 'random' creates n random sensors, while
        'orthogonal creates two sensors orthogonal to the walls

    returns: 2xn array, where each column holds a sensor vector.
    '''
    sensors = np.array((n,2))
	
    if direction == 'orthogonal': 
        assert(n == 2)
		
        return np.array([[1,0], [0,1]])
		
    elif direction == 'random': 
        angles = np.random.uniform(0, 2*np.pi, n)
        sensors = np.array([np.cos(angles), np.sin(angles)])
		 
        return sensors
		 
    else:
        print('Wrong directions keyword')
		
def sonar(pos, sensors, width=ROOMWIDTH, length=ROOMLENGTH): 
    '''
    pos: array or list, signifies the position from which distances are computed
    sensors: 2xn array of sensors as created by generate_sensors

    returns: nx2 array, where the first column holds the distaces according to 
        each of the sensors, the second one the index of the wall the distance is measured to.
    '''
    #creating the points for the intersection
    p1 = tuple(pos)
    #tuple
    sensors_p2 = [(pos[0]+sensors[0, i],pos[1]+sensors[1, i]) for i in range(sensors.shape[1])] 
    #list of tuples
    walls_p3 = [(0,0), (0,0), (0,length), (width,length)] 
    #list of tuples
    walls_p4 = [(width,0), (0,length), (width,length), (width,0)]
    #list of tuples
	
    #distances = np.zeros((len(sensors_p2), len(walls_p3)))
    #old distances array

    distances = np.zeros((sensors.shape[1], 2))
    #one distance per sensor, plus the integer corresponding to the wall the 
    #distance is measured from

    for i in range(len(sensors_p2)):
        target_walls = []
        if sensors[0,i] > 0:
            target_walls.append(WALLSDICT['west'])
        if sensors[0,i] < 0:
            target_walls.append(WALLSDICT['east'])
        if sensors[1,i] > 0:
            target_walls.append(WALLSDICT['north'])
        if sensors[1,i] < 0:
            target_walls.append(WALLSDICT['south'])
        #We check which walls lie in the direction of the sensor
        
        distances_temp = np.zeros((len(target_walls), 2))
        
        for k, j in enumerate(target_walls):	
            intersect = geo.getIntersectPoint(p1, sensors_p2[i], walls_p3[j], walls_p4[j])

            if not intersect: 
                distances_temp[k] = [np.nan, j]
            else:
                dist = np.linalg.norm(np.array(pos) - np.array(intersect))
                distances_temp[k] = [dist, j]
        #compute the distances to all of these walls

        distances[i,0] = np.nanmin(distances_temp[:,0])
        distances[i,1] = distances_temp[np.nanargmin(distances_temp[:,0]),1]
        #take the minimum

        ### ### ### old version ### ### ###
        #for j in range(len(walls_p3)):
        #	
        #    print(p1, sensors_p2[i], walls_p3[j], walls_p4[j])
        #    intersect= geo.getIntersectPoint(p1, sensors_p2[i], walls_p3[j], walls_p4[j])
        #
        #    if not intersect: 
        #        distances[i, j]= -1
        #    else:
        #        dist= np.linalg.norm(np.array(pos)-np.array(intersect))
        #        distances[i, j]= dist
    
    return distances	
	
	
		









