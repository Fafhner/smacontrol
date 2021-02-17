import numpy as np
from smacontrol import Motors
import matplotlib.pyplot as plt

"""Simulation parameters"""
N = 1  # Simulation time's end
dt = 0.0001  # Sampling time
max_len = int(np.round(N / dt))  # Simulation's vectors' length
time = np.arange(0, N, dt)  # Time vector
Udc = 400
Tl = 20

R = 3
L = 0.04
J = 0.5
B = 0.05
k = 2.1
p = 1
M = Motors.BLDC(R=R, L=L, J=J, mu=B, ke=k, kt=k, p=p, dt=dt, len=max_len)


"""Simulation"""

for it in range(0, max_len):
    G = Motors.BLDC_ideal_commutator(M.now[0:3], M.back_Emf.now[0:3])
    Ua, Ub, Uc = Motors.BLDC_ideal_inverter(Udc=Udc, G=G)
    M.simulate(inputs=[[Ua - Ub], [Ub - Uc], [Tl]], dt=dt)



"""Plotting"""
params = r"Motor parameters: R=" + str(R) + ", L=" + str(L) + ", B=" + str(B) + ", J=" + str(J) + ", k=" + str(
    k) + ", p=" + str(p)
params2 = r" $ U{dc}=$" + str(Udc) + r", $T_l=$" + str(Tl) + ", dt= " + str(dt)



plt.plot(time, M[0, :], time, M[1, :], time, M[2, :])
plt.legend([r"$i_A$", r"$i_B$", r"$i_C$"])
plt.xlabel("Time [s]")
plt.ylabel("Current [A]")
plt.title(r"BLDC currents $i_A$, $i_B$, $i_C$" + "\n" + params + "\n" + params2, fontsize=10)
plt.grid()
plt.show()


plt.plot(time, M[3, :])
plt.xlabel("Time [s]")
plt.ylabel(r"Speed of rotor $\omega$ [$\frac{1}{rad}$]")
plt.title(r"BLDC rotation speed of rotor $\omega$" + "\n" + params + "\n" + params2, fontsize=10)
plt.grid()
plt.show()

plt.plot(time, M[4, :])
plt.xlabel("Time [s]")
plt.ylabel(r"Rotor angle $\theta_m$ [$rad$]")
plt.title(r"BLDC mechanical angle $\theta_m$ of rotor" + "\n" + params + "\n" + params2, fontsize=10)
plt.grid()
plt.show()


# UNCOMMENT TO SAVE TO .CSV
# """Save to file"""
# 1 - ia | 2 - ib | 3 - ic | 4 - speed | 5 - angle"
#np.savetxt("test_bldc_output.csv", X=(M[0, :], M[1, :], M[2, :], M[3, :], M[4, :]),  delimiter=';')
#  | 1 - Te | 2 - ea |"
#np.savetxt("test_bldc_te_ea.csv", X=(M.torque[0, :], M.back_Emf[3, :]), delimiter=';')
