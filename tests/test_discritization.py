from smacontrol import DiscreteTime as dtime
import numpy as np

# Parameters
A = np.asanyarray([[1, 2], [1, 1]])
B = np.asanyarray([[1], [-1]])
C = np.asanyarray([[2, 0]])
D = np.asanyarray([[5]])
T = 0.5

# Discerete matrices
Ad, Bd, Cd, Dd = dtime.discretization_tustin([A, B, C, D], T=T)


print('Ad matrix:\n', Ad)
print('Bd matrix:\n', Bd)
print('Cd matrix:\n', Cd)
print('Dd matrix:\n', Dd)
