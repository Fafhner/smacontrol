import numpy as np
from smacontrol import Motors
import matplotlib.pyplot as plt

"""Simulation parameters"""
N = 0.8  # Simulation time's end
dt = 0.001  # Sampling time
max_len = int(N / dt)  # Simulation's vectors' length
time = np.arange(0, N, dt)  # Time vector
Umax = 300
# Voltage
U = Umax * np.ones((1, max_len))  # Input vector u=1
# Load
Tl_start = 0.45
Tl = 40*np.ones((1, max_len))
Tl[0, :int(Tl_start/dt)] = 0

R, L, B, J, k, fi = 3, 10e-3, 0.4, 0.11, 2.23, 1
M = Motors.DCMotor(R=R, L=L, mu=B, J=J, k=k, fi=fi, dt=dt, len=max_len)


"""Simulation"""
for t in range(0, max_len):
    M.simulate(inputs=[U[0, t], Tl[0, t]], dt=dt)


"""Plotting"""
plt.plot(time, M.current[:], time, Tl[0, :])
plt.xlabel("Time [s]")
plt.ylabel(r"Armature current $I_a$ [A]")
plt.legend(["Load", "Current"])
plt.title("Seprately excited DC motor simulation\n" +
          f"Parameters: Ra={R}, La={L}, B={B}, J={J}, k={k}, fi={fi}")
plt.grid()
plt.show()

plt.plot(time, M.speed[:])
plt.xlabel("Time [s]")
plt.ylabel(r"Rotor speed $\omega$ [$\frac{rad}{s}$]")
plt.title("Seprately excited DC motor simulation\n" +
          f"Parameters: Ra = {R}, La = {L}, B = {B}, J = {J}, k = {k}, fi = {fi}")
plt.grid()
plt.show()

plt.plot(time, M.torque[:])
plt.xlabel("Time [s]")
plt.ylabel(r"Electromagnetic torque $T_e$ [N]")
plt.title("Seprately excited DC motor simulation \n" +
          f"Parameters: Ra = {R}, La = {L}, B = {B}, J = {J}, k = {k}, fi = {fi}")
plt.grid()
plt.show()

# UNCOMMENT TO SAVE TO .CSV
# """Save to file"""
#np.savetxt("test_dc_1_data.csv", X=(M.current[:], M.speed[:], M.torque[:]), delimiter=';')
