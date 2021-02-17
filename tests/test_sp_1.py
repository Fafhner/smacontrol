import numpy as np
from smacontrol import SmithPredictor as SP, Models
import matplotlib.pyplot as plt

"""Simulation parameters"""
N = 50  # Simulation time's end
dt = 0.1  # Sampling time
max_len = int(N / dt)  # Simulation's vectors' length
time = np.arange(0, N, dt)  # Time vector
u = np.ones((1, max_len))  # Input vector u=1
v = -1*np.ones((1, max_len))  # Disturbance vector v=-1, for t>=20
v_start = int(20 / dt)
v[0, :v_start] = 0

"""Open loop - step response"""
# Process model P(s) =  1/(1s+1)e^-5 to state-space
A, B, C, D = [[-1]], [[1]], [[1]], [[0]]
d = int(5 / dt)
Plant_openloop = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Delay_openloop = Models.Delay(d=d, dt=dt, len=max_len)

"""Smith Predictor"""
# Process model P(s) = 1/(1s+1)e^-5 to state-space
Plant = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Delay = Models.Delay(d=d, dt=dt, len=max_len)
# Model Pmo(s) = 1/(1s+1)
Model = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
# Regulator C(s) = (1s + 1)/(0.5s)
Ar, Br, Cr, Dr = [[0]], [[1]], [[2]], [[2]]
Reg = Models.StateSpace(ABCD=[Ar, Br, Cr, Dr], dt=dt, len=max_len)
# Output memory
Mem = Models.Memory(in_len=1, dt=dt, len=max_len)
# SmithPredictor structure
SmithPred = SP.SmithPredictor(reg=Reg, model=Model, delay=d, len=max_len)

"""Simulation"""
for it in range(0, max_len):
    # Closed loop
    Mem.simulate(inputs=Delay.now + v[0, it])
    SmithPred.simulate(inputs=[[u[0, it]], [Mem.now]], dt=dt)
    Plant.simulate(inputs=SmithPred.now, dt=dt)
    Delay.simulate(inputs=Plant.now, dt=dt)

    # Open-loop simulation
    Plant_openloop.simulate(inputs=u[0, it], dt=dt)
    Delay_openloop.simulate(inputs=Plant_openloop.now, dt=dt)

"""Plotting"""

plt.plot(time, u[0, :], '--')
plt.plot(time, Delay_openloop[:] + v[0, :])
plt.grid()
plt.legend(["Input",  "SP - Open loop - step response."])
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{1}}{{s+1}}e^{{-5s}}$ with open loop")
plt.show()

plt.plot(time, u[0, :], '--')
plt.plot(time, Mem[:])
plt.grid()
plt.legend(["Input", "SP - Plant output"])
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{1}}{{s+1}}e^{{-5s}}$ with closed loop"
          + "\n" + fr"model Pm(s)=$\frac{{1}}{{s+1}}e^{{-5s}}$, controller C(s)=$\frac{{s+1}}{{0.5s}}$")
plt.show()

# UNCOMMENT TO SAVE TO .CSV
#"""Save to file"""
# | 1 - Open-loop | 2 - Closed-loop
#np.savetxt("test_sp_1_data.csv", X=(Delay_openloop[0, :] - v[0, :], Mem[0, :]), delimiter=';')