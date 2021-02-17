from smacontrol import Models
import numpy as np


class PID(Models.Memory):
    """
        Parallel realization of PID controller C(s) = Kp*(1 + 1 / (Ti*s) + Td*s/(1+(Td/N)*s)) = Cp(s)*Ci(s)*Cd(s)
        in discrete time.
        If N=inf by default, then derivative term is ideal and Cd(s) = Td*s
        If alpha or beta are given, then controller becomes feedforward 2-degree of freedom controller.
    """

    def __init__(self, Kp=1, Ti=0, Td=0, dt=1, N=float('inf'), alpha=None, beta=None, **opt):
        """
        Arguments:
        ----------
        Kp : `float`
            Proportional gain.
            Default ``Kp``=1.
        Ti :  `float`
            Integral time constant.
            Default ``Ti``=0.
        Td : `float`
            Derivative time constant.
            Default ``Td``=0.
        dt : `float`
            Sampling period.
            Default ``dt``=1
        N : `float`
            Derivative filter coefficient
            Default ``N``=float('inf')
        alpha : `float`
            Alpha coefficient for two-degree of freedom controller.
            Default ``alpha``=None
        beta : `float`
            Alpha coefficient for two-degree of freedom controller.
            Default ``alpha``=None
        """
        self.P_block = Models.Gain(Kp=Kp, dt=dt, **opt) if Kp else None
        self.I_block = Models.Integral(Ki=Kp / Ti, dt=dt, **opt) if Ti else None
        self.D_block = None
        self.P2_block = None
        self.D2_block = None
        if Td:
            self.D_block = Models.Derivative(Kd=Td * Kp, dt=dt, **opt) if N == float('inf') \
                else Models.StateSpace(ABCD=[[[-N / Td]], [[1]], [[- N ** 2 / Td]], [[N]]], dt=dt, K=Kp, **opt)

            if alpha: self.P2_block = Models.Gain(Kp=Kp * alpha, **opt)
            if beta:
                self.D2_block = Models.Derivative(Kd=Kp * Td * beta, dt=dt, **opt) if N == float('inf') \
                    else Models.StateSpace(ABCD=[[[-N / Td]], [[1]], [[- N ** 2 / Td]], [[N]]], dt=dt, K=Kp * beta, **opt)

        super().__init__(in_len=1, name='PID_Parallel', **opt)

    def simulate(self, inputs, dt=None):
        """Simulation of a block.

         Arguments:
         ----------
         inputs : `float`, `numpy.ndarray`
             Input to the controller as array ['e'] for 1-degree controller or ['e', 'u'] for 2-degree controller,
             where 'u' is a setting value, 'e' is an error 'e = yplant - u' with plant output 'yplant'.
         dt : `float`

         Return:
         -------
         y : `float`
             Returns last value from regulator.
         """
        inputs = Models.convert_2dim_array(inputs).reshape(-1, 1)

        if inputs.shape[0] == 1:
            e = inputs
            u = np.zeros((1, 1))
        elif inputs.shape[0] == 2:
            e, u = inputs
        else:
            raise Exception("PID: Expected max. 2 inputs. Got {}".format(inputs.shape))


        yp, yi, yd, yalfa, ybeta = np.zeros((5, 1, 1))

        if self.P_block: yp = self.P_block.simulate(inputs=e, dt=dt)
        if self.I_block: yi = self.I_block.simulate(inputs=e, dt=dt)
        if self.D_block: yd = self.D_block.simulate(inputs=e, dt=dt)
        if self.P2_block: yalfa = self.P2_block.simulate(inputs=u, dt=dt)
        if self.D2_block: ybeta = self.D2_block.simulate(inputs=u, dt=dt)
        y = yp + yi + yd + yalfa + ybeta
        super().simulate(y, dt=dt)

        return y


class SerialPID(Models.Memory):
    """
        Serial realization of PID controller C(s) = Kp*(1 + 1/(Ti*s))((1 + Td*s)/(1 + (Td/N)*s)) = Cp(s)*Ci(s)*Cd(s)
        If N=inf by default, then derivative term is ideal and Cd(s) = (1+Td*s).
    """

    def __init__(self, Kp=0, Ti=0, Td=0, dt=1, N=float('inf'), **opt):
        """
       Arguments:
       ----------
       Kp : `float`
           Proportional gain.
           Default ``Kp``=1.
       Ti :  `float`
           Integral time constant.
           Default ``Ti``=0.
       Td : `float`
           Derivative time constant.
           Default ``Td``=0.
       dt : `float`
           Sampling period.
           Default ``dt``=1
       N : `float`
           Derivative filter coefficient
           Default ``N``=float('inf')
       """
        self.P_block = Models.Gain(Kp=Kp, dt=dt, **opt) if Kp else None
        self.I_block = Models.Integral(Ki=1 / Ti, dt=dt, **opt) if Ti else None
        self.D_block = None
        if Td:
            self.D_block = Models.Derivative(Kd=Td, dt=dt, **opt) if N == float('inf') \
                else Models.StateSpace(ABCD=[[[-N / Td]], [[1]], [[- N ** 2 / Td]], [[N]]], dt=dt, **opt)

        super().__init__(in_len=1, name='PID_Serial', **opt)

    def simulate(self, inputs, dt=None):
        """Simulation of a block.

        Arguments:
        ----------
        e : `float`, `numpy.ndarray`
            Input to the controller as error e=u-y, where u is a setting value, y a return value from a process.
        time : `int`
            Number of simulation iterations.
            Default ``time``=1

        Return:
        -------
        y : `float`
            Returns last value of simulation
        """
        e = Models.convert_2dim_array(inputs).reshape(-1, 1)

        yd = self.D_block.simulate(inputs=e, dt=dt) + e if self.D_block else e
        yp = self.P_block.simulate(inputs=yd, dt=dt) if self.P_block else yd
        yi = self.I_block.simulate(inputs=yp, dt=dt) + yp if self.I_block else yp

        super().simulate(yi)

        return yi
