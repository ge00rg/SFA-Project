import numpy as np
import moving_bat as mb
import matplotlib.pyplot as plt
import sfa_tasks as sfa

def nr_1(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    mb.plot_trajectory(traj)

def nr_2(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)

    n = np.random.randint(1,5)
    sensors = mb.generate_sensors(n=n)
    data = mb.generate_data(traj, sensors)
    
    for i in range(data.shape[1]):
        plt.plot(data[:,i], label='sensor {}'.format(i+1))

    plt.xlabel('time')
    plt.ylabel('distance')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5,-.15), ncol=data.shape[1])
    plt.show()

def nr_3(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):

    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)

    sensors = mb.generate_sensors(n=2, direction='orthogonal')
    sen = mb.sensory_data(traj, sensors)

    data = mb.generate_data(traj, sen)
    slow = sfa.do_sfa(data)

    mesh = sfa.mesh(traj, slow)

    plt.imshow(mesh)
    plt.show()

def nr_4():
    pass

def nr_5():
    pass

def nr_6():
    pass

def nr_7():
    pass

def nr_8():
    pass

############ testing grounds #################

nr_3(quadratic=True)
