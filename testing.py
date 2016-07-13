import moving_bat as mb
import numpy as np

mb.plot_trajectory()

sen1=mb.generate_sensors()
sen2=mb.generate_sensors(2,'orthogonal')
#sen3=mb.generate_sensors(3,'orthogonal')
sen4=mb.generate_sensors(5)
sen5=mb.generate_sensors(3,'test')

print('arg=None', sen1)
print('arg=2, orthogonal', sen2)
print(np.linalg.norm(sen4[:,1]))
#print('arg=3, orthogonal', sen3)
print('arg=5', sen4)
print('arg=3, test', sen5)


