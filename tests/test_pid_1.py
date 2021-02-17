import numpy as np
import matplotlib.pyplot as plt
from smacontrol import PID, Models

"""Simulation parameters"""
N = 100  # Simulation time's end
dt = 0.01  # Sampling time
max_len = int(N / dt)  # Simulation's vectors' length
time = np.arange(0, N, dt)  # Time vector
u = np.ones((1, max_len))  # Input vector u=1
input_on_time = 10  # Start time of input
u[0, :int(input_on_time / dt)] = 0  # Input u = 0 if t < input_on_time

"""State-Space model of inertia P(s) = 2/(50s+1)"""
A = [[-1 / 50]]
B = [[1]]
C = [[2 / 50]]
D = [[0]]
Plant_parallel = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Plant_serial = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Plant_open = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
"""Parallel and serial PID objects"""
Kp = 10
Ti = 2
Td = 0.2
Reg_PID_parallel = PID.PID(Kp=Kp, Ti=Ti, Td=Td, dt=dt, len=max_len)

Kps = Kp / 2 * np.sqrt(1 + 4 * Td / Ti)
Tis = Ti / 2 * np.sqrt(1 + 4 * Td / Ti)
Tds = Ti / 2 * np.sqrt(1 - 4 * Td / Ti)
Reg_PID_serial = PID.SerialPID(Kp=Kps, Ti=Tis, Td=Tds, dt=dt, len=max_len)


"""Simulation"""
for it in range(0, max_len):
    t = time[it]
    # Parallel
    Reg_PID_parallel.simulate(u[0, it] - Plant_parallel.now, dt=dt)
    Plant_parallel.simulate(Reg_PID_parallel.now, dt=dt)

    # Serial
    Reg_PID_serial.simulate(u[0, it] - Plant_serial.now, dt=dt)
    Plant_serial.simulate(Reg_PID_serial.now, dt=dt)

    # Open loop
    Plant_open.simulate(u[0, it], dt=dt)



"""Plot"""
# Plot setting value, plant output for parallel PID, plant output for open loop
plt.plot(time, u[0, :], '--')
plt.plot(time, Plant_parallel[0, :])
plt.plot(time, Plant_open[0, :], '--')
plt.grid()
plt.legend(["Input", "Plant output \n- closed loop, with PID", "Plant output \n- open loop, without PID"])
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{2}}{{50s+1}}$"
          + "\n and parallel PID with param."
          + fr" $K_p$={Kp}, $T_i$={Ti}, $T_d$={Td}")
plt.show()

# Plot setting value, plant output for serial PID, plant output for open loop
plt.plot(time, u[0, :], '--')
plt.plot(time, Plant_serial[0, :])
plt.plot(time, Plant_open[0, :], '--')
plt.grid()
plt.legend(["Input", "Plant output \n- closed loop, with PID", "Plant output \n- open loop, without PID"])
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{2}}{{50s+1}}$"
          + "\n and serial PID with param."
          + fr" $K_p$={np.around(Kps, 2)}, $T_i$={np.around(Tis, 2)}, $T_d$={np.around(Tds, 2)}")
plt.show()

# Plot plant output for parallel and serial PID
plt.plot(time, Plant_parallel[0, :])
plt.plot(time, Plant_serial[0, :], '--')
plt.grid()
plt.legend(["Plant output \n- parallel PID", "Plant output \n- serial PID"])
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{2}}{{50s+1}}$"
          + "\n Comparison between parallel and serial PID")
plt.show()


# UNCOMMENT TO SAVE TO .CSV
# """Save to file"""
# | 1 - Plant_open | 2 - PID parallel | 3 - PID serial |
#np.savetxt("test_PID_1_data.csv", X=[u[0, :], Plant_open[0, :], Plant_parallel[0, :], Plant_serial[0, :], ], delimiter=';')
