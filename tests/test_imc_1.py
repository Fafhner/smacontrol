import numpy as np
from smacontrol import IMC, Models
import matplotlib.pyplot as plt

"""Simulation parameters"""
N = 100  # Simulation time's end
dt = 0.01  # Sampling time
max_len = int(N / dt)  # Simulation's vectors' length
time = np.arange(0, N, dt)  # Time vector
u = np.ones((1, max_len))  # Input vector u=1
input_on_time = 20  # Start time of input
u[0, 0:int(input_on_time / dt)] = 0  # Input u = 0 if t < input_on_time

"""
Open loop - step response
"""
# Process P(s) = 12/(12s+1)e^-2 to state-space
A, B, C, D = [[-1 / 12]], [[1]], [[1]], [[0]]
d = int(2 / dt)
Plant_openloop0 = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Delay_openloop0 = Models.Delay(d=d, dt=dt, len=max_len)

"""
******
*** IMC 1 - model with delay
******
"""
# Process P(s) = 12/(12s+1)e^-2s to state-space
Plant1 = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)

Delay1 = Models.Delay(d=d, dt=dt, len=max_len)
# Model Pm(s) = 12/(12s+1)e^-2s
Plant_model1 = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
# Regulator Q(s) = 12s+1/6s+12
Ar1, Br1, Cr1, Dr1 = [[-2]], [[1]], [[-23 / 6]], [[2]]
Reg1 = Models.StateSpace(ABCD=[Ar1, Br1, Cr1, Dr1], dt=dt, len=max_len)
# IMC
IMC1 = IMC.IMC(reg=Reg1, model=Plant_model1, delay=d, len=max_len)


"""
******
*** IMC 2 - model delay approx. with Pade.
******
"""
# Process P(s) = 12/(12s+1)e^-2s to state-space
Plant_pade2 = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Delay_pade2 = Models.Delay(d=d, dt=dt, len=max_len)
# Model Pm(s) = -12s+12/(12s^2+13s+1)
Am2, Bm2, Cm2, Dm2 = [[0, 1], [-1 / 12, -13 / 12]], [[0], [1]], [[1, -1]], [[0]]
Plant_model_pade2 = Models.StateSpace(ABCD=[Am2, Bm2, Cm2, Dm2], dt=dt, len=max_len)
# Regulator Q(s)=(12s^2+13s+1)/(3s^2+24s)
Ar2, Br2, Cr2, Dr2 = [[0, 1], [-4, -4]], [[0], [1]], [[-47 / 3, -35 / 3]], [[4]]
Reg_pade2 = Models.StateSpace(ABCD=[Ar2, Br2, Cr2, Dr2], dt=dt, len=max_len)
IMC_pade2 = IMC.IMC2(reg=Reg_pade2, model=Plant_model_pade2, len=max_len)
Plant_model_pade2.initial_state()

"""
******
*** Classic control - model delay approx. with Pade.
******
"""
# Process model P(s) = 12/(12s+1)e^-2s to state-space
Plant_classic3 = Models.StateSpace(ABCD=[A, B, C, D], dt=dt, len=max_len)
Delay_classic3 = Models.Delay(d=d, dt=dt, len=max_len)
# Regulator C(s)=(12s^2+13s+1)/(3s^2+24s)
Ar3, Br3, Cr3, Dr3 = [[0, 1], [0, -8]],  [[0], [1]], [[1 / 3, -83 / 3]], [[4]]
Reg_classic3 = Models.StateSpace(ABCD=[Ar3, Br3, Cr3, Dr3], dt=dt, len=max_len)



"""Simulation"""
yr2, yp2, yd2 = 0, 0, 0
yr3, yp3, yd3 = 0, 0, 0
for it in range(0, max_len):
    # IMC - model with delay
    IMC1.simulate(inputs=[u[0, it], Delay1.now], dt=dt)
    Plant1.simulate(inputs=IMC1.now, dt=dt)
    Delay1.simulate(inputs=Plant1.now, dt=dt)

    # IMC - model delay approx. with Pade
    yr2 = IMC_pade2.simulate(inputs=[u[0, it], yd2], dt=dt)
    yp2 = Plant_pade2.simulate(inputs=yr2, dt=dt)
    yd2 = Delay_pade2.simulate(inputs=yp2, dt=dt)

    # Classic
    yr3 = Reg_classic3.simulate(inputs=u[0, it] - yd3, dt=dt)
    yp3 = Plant_classic3.simulate(yr3, dt=dt)
    yd3 = Delay_classic3.simulate(inputs=yp3, dt=dt)

    # Open loop
    Plant_openloop0.simulate(inputs=u[0, it], dt=dt)
    Delay_openloop0.simulate(inputs=Plant_openloop0.now, dt=dt)

"""Plotting"""

plt.plot(time, u[0, :], '--')
plt.plot(time, Delay_openloop0[:])
plt.legend(["Input", "Open loop - step response."])
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{12}}{{12s+1}}e^{{-2}}$ with open loop")
plt.show()

plt.plot(time, u[0, :], '--')
plt.plot(time, Delay1[:])
plt.legend(["Input", "IMC - Plant output \n- without Pade approx."])
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{12}}{{12s+1}}e^{{-2}}$ with closed loop"
          + "\n" + fr"model Pm(s)=$\frac{{12}}{{12s+1}}e^{{-2}}$, controller Q(s)=$\frac{{12s+1}}{{6s+12}}$", fontsize=10)
plt.show()

plt.plot(time, u[0, :], '--')
plt.plot(time, Delay_pade2[:])
plt.legend(["Input", "IMC - Plant output \n- with Pade approx."])
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{12}}{{12s+1}}e^{{-2}}$ with closed loop"
          + "\n" + fr"model Pm(s)=$\frac{{-12s+12}}{{12s^2+13s+1}}$, controller Q(s)=$\frac{{12s^2 + 13 s + 1}}{{3 "
                   fr"s^2 + 12s + 12}}$", fontsize=10)
plt.show()

plt.plot(time, u[0, :], '--')
plt.plot(time, Delay_classic3[:])
plt.legend(["Input", "Classical control \n- with Pade approx."])
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title(fr"Simulation of process P(s)=$\frac{{12}}{{12s+1}}e^{{-2}}$ with closed loop"
          + "\n" + fr"controller C(s)=$\frac{{12s^2+13s+1}}{{3s^2+24s}}$", fontsize=10)
plt.show()


plt.plot(time, Delay1[0, :])
plt.plot(time, Delay_pade2[0, :], '--')
plt.plot(time, Delay_classic3[0, :], ':')
plt.legend(["Dead time", "Pade approx of dead time", "Classical control"])
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Plant output y(t)")
plt.title("Comparison between: Model with dead time vs \n model with delay approx. vs classical control")
plt.show()

# UNCOMMENT TO SAVE TO .CSV
# """Save to file"""
# | 1 - open loop | 2-Output w/o delay approx | 3 - Output with delay approx | 4 -classical controler
#np.savetxt("test_imc_1_data.csv", X=(Delay_openloop0[:], Delay1[:], Delay_pade2[:], Delay_classic3[:]), delimiter=';')
