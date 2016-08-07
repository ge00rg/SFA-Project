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
   

def nr_4(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    sen = mb.generate_sensors(n=3, direction='random')
    mb.plot_sensors(sen)
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=1, whiten=True, svd=False)
    #todo: at the moment n must be equal to out_dim because of dim-mismatch

    sfa.mesh(sen, data, flow)
    
def nr_4_plot_square(width=SQUARE_SIDE, length=SQUARE_SIDE, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED, savestring=" "):
#    mb.ROOMLENGTH=7
#    mb.ROOMWIDTH=7
    sen_numb=[5, 11, 23]
    traj = mb.make_trajectory(width=SQUARE_SIDE, length=SQUARE_SIDE, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED)
    plt.figure(figsize=(9,9))
    for i,s in enumerate(sen_numb): 
        sen = mb.generate_sensors(n=s, direction='random')
        data = mb.generate_data(traj, sen, width=SQUARE_SIDE, length=SQUARE_SIDE)
        flow = sfa.train_sfa(data, poly_exp=1, whiten=True, svd=True)
        grid_plot= sfa.mesh(sen, data, flow, spacing=0.1, width=SQUARE_SIDE, length=SQUARE_SIDE, ret_dim=5, ica=False, icadim=2, draw=False, save=False, savestring='')
       
        plt.subplot(3,3,i*3+1)
        for j in range(sen.shape[1]):
            plt.plot([0, sen[0,j]], [0,sen[1,j]])
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.axis("off")
        plt.title("sensor directions")
        #plt.tick_params(axis='both', which='both', bottom='off', top='off',labelbottom='off') 
        #plt.tick_params(axis='y', which='both', bottom='off', top='off',labelbottom='off') 
            
        plt.subplot(3,3,i*3+2)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1) 
        #plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,0], interpolation='none', origin='lower')
        plt.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off') 
        plt.title("component= 1")
        #plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
            
        plt.subplot(3,3,i*3+3)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1)
        #plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,1], interpolation='none', origin='lower')
        plt.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off') 
        plt.title("component= 1")
        #plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
#        plt.subplot(3,4,i*4+4)
#        rng = np.max([length, width])
#        yticks = np.arange(0, width, 1)
#        xticks = np.arange(0, length, 1)
#        plt.xticks(np.arange(0, length*10,10),xticks)
#        plt.yticks(np.arange(0, width*10,10),yticks)
#        plt.imshow(grid_plot[:,:,1], interpolation='none', origin='lower')
#        plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
    plt.tight_layout()
    plt.savefig(savestring)



def nr_5(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    sen = mb.generate_sensors(n=3, direction="random")
    mb.plot_sensors(sen)
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa(data, poly_exp=3, whiten=True, svd=True)
    #fig_title="sen_20_pol_exp_7"
    sfa.mesh(sen, data, flow)
    
def nr_5_iterable(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED, savestring=" "):
    poly_numb=[3, 5, 7]
    print(width, length)
    print(mb.ROOMWIDTH, mb.ROOMLENGTH)
        
    traj = mb.make_trajectory()
    sen = mb.generate_sensors(n=5, direction="random")
    data = mb.generate_data(traj, sen)
    
    plt.figure(figsize=(15,6))    
    for i,p in enumerate(poly_numb): 
        print(p)

        flow = sfa.train_sfa(data, poly_exp=p, whiten=True, svd=True)
        grid_plot= sfa.mesh(sen, data, flow, spacing=0.1, width=width, length=length, ret_dim=5, ica=False, icadim=2, draw=False, save=False, savestring='')
        print(grid_plot.shape)
       
        plt.subplot(3,3,i*3+1)
        plt.title("degree= "+str(p)+ " component= 1", fontsize=10)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1) 
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,0], interpolation='none', origin='lower')
        #plt.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off') 
        #plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
        plt.subplot(3,3,i*3+2)
        plt.title("degree= "+str(p)+ " component= 2", fontsize=10)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1) 
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,1], interpolation='none', origin='lower')
        #plt.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off') 
        #plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
            
        plt.subplot(3,3,i*3+3)
        plt.title("degree= "+str(p)+ " component= 3", fontsize=10)
        rng = np.max([length, width])
        yticks = np.arange(0, width, 1)
        xticks = np.arange(0, length, 1)
        plt.xticks(np.arange(0, length*10,10),xticks)
        plt.yticks(np.arange(0, width*10,10),yticks)
        plt.imshow(grid_plot[:,:,2], interpolation='none', origin='lower')
        #plt.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off') 
        

    plt.tight_layout()
    plt.savefig(savestring)

   
    
def nr_6(width=mb.ROOMWIDTH, length=mb.ROOMLENGTH, minspeed=mb.MINSPEED, maxspeed=mb.MAXSPEED, out=3, savestring=" "):
    traj = mb.make_trajectory(width=width, length=length, minspeed=minspeed, maxspeed=maxspeed)
    sen = mb.generate_sensors(n=5, direction="random")
    data = mb.generate_data(traj, sen)
    flow = sfa.train_sfa_with_ica(data, poly_exp=11, o_dim=out)
    sfa.do_sfa_ica(sen, flow, save=True, savestring=savestring) 

    


def nr_7():
    pass

def nr_8():
    pass

############ testing grounds #################

nr_6(out=4, savestring="ICA_sen_5_rand_poly_11_out_3_rect")



