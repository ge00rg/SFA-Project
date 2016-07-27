import moving_bat as mb
import sfa_tasks as sfa
import matplotlib.pyplot as plt

traj = mb.make_trajectory()
sensors = mb.generate_sensors()
sen = mb.sensory_data(traj, sensors)

print(sen.shape)

out = sfa.do_sfa(sen)

plt.plot(out)
plt.show()
