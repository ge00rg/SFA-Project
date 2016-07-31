import numpy as np
import moving_bat as mb
import matplotlib.pyplot as plt
import sfa_tasks as sfa

traj = mb.make_trajectory()

sensors = mb.generate_sensors()
sen = mb.sensory_data(traj, sensors)

data = mb.generate_data(traj, sen)

slow = sfa.do_sfa(data)

mesh = sfa.mesh(traj, slow)

plt.figure(1)
plt.plot(sen[:,0])
plt.plot(sen[:,1])
plt.xlim(-10,mb.T+10)

plt.figure(2)
plt.plot([0, sensors[0,0]], [0, sensors[1,0]])
plt.plot([0, sensors[0,1]], [0, sensors[1,1]])

plt.figure(3)
plt.imshow(mesh)

mb.plot_trajectory(traj)

plt.show()

