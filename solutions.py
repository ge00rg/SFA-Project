import numpy as np
import moving_bat as mb
import matplotlib.pyplot as plt
import sfa_tasks as sfa

SQUARE_SIDE=7

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
    sen = mb.generate_sensors(n=2, direction='orthogonal')
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=1)

    sfa.mesh(sen, data, flow)
    
def nr_3_plot(width=SQUARE_SIDE, length=SQUARE_SIDE, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    sen_numb=[3,5,15]
    traj = mb.make_trajectory(width=SQUARE_SIDE, length=SQUARE_SIDE, minspeed=minspeed, maxspeed=maxspeed)
    plt.figure(figsize=(9,9))
    for i,s in enumerate(sen_numb): 
        sen = mb.generate_sensors(n=s, direction='random')
        data = mb.generate_data(traj, sen, width=SQUARE_SIDE, length=SQUARE_SIDE)
        flow = sfa.train_sfa(data, poly_exp=1, whiten=True)
        grid_plot= sfa.mesh(sen, data, flow, spacing=0.1, width=SQUARE_SIDE, length=SQUARE_SIDE, ret_dim=5, ica=False, icadim=2, draw=False, save=False, savestring='')

       
        plt.subplot(3,3,i*3+1)
        for j in range(sen.shape[1]):
            plt.plot([0, sen[0,j]], [0,sen[1,j]])
        plt.xlim(-2,2)
        plt.ylim(-2,2)
            
        plt.subplot(3,3,i*3+2)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1) 
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,0], interpolation='none', origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
            
        plt.subplot(3,3,i*3+3)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1)
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,1], interpolation='none', origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
    plt.show()

def nr_3_plot_rect(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    sen_numb=[3,5,15]
    traj = mb.make_trajectory()
    plt.figure(figsize=(9,9))
    for i,s in enumerate(sen_numb): 
        sen = mb.generate_sensors(n=s, direction='random')
        data = mb.generate_data(traj, sen)
        flow = sfa.train_sfa(data, poly_exp=1, whiten=True)
        grid_plot= sfa.mesh(sen, data, flow, spacing=0.1, ret_dim=5, ica=False, icadim=2, draw=False, save=False, savestring='')

       
        plt.subplot(3,3,i*3+1)
        for j in range(sen.shape[1]):
            plt.plot([0, sen[0,j]], [0,sen[1,j]])
        plt.xlim(-2,2)
        plt.ylim(-2,2)
            
        plt.subplot(3,3,i*3+2)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1) 
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,0], interpolation='none', origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
            
        plt.subplot(3,3,i*3+3)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1)
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,1], interpolation='none', origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
    plt.show()


def nr_4(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    sen = mb.generate_sensors(n=3, direction='random')
    mb.plot_sensors(sen)
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=1, whiten=True, svd=False)
    #todo: at the moment n must be equal to out_dim because of dim-mismatch

    sfa.mesh(sen, data, flow)

def nr_5(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
#    sen = mb.generate_sensors(n=3, direction="random")
#    f=open("sensors.txt", "w")
#    f.write(str(sen))
#    f.close()
    sen=np.array([[-0.95377798, -0.4048993,  -0.57044942],
 [ 0.30051217, -0.91436128, -0.82133274]])
    mb.plot_sensors(sen)
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=7, whiten=True, svd=True)
    fig_title="sen_3_pol_exp_7"
    sfa.mesh(sen, data, flow, savestring=fig_title)

def nr_6(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    sen = mb.generate_sensors()
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=7, whiten=True, svd=True, ica=True, icadeg=None)

    sfa.mesh(sen, data, flow)

    pass

def nr_7():
    pass

def nr_8():
    pass

############ testing grounds #################

nr_3()
