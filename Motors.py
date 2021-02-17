import numpy as np
from smacontrol import Models

"""

DC motor

"""


class DCMotor(Models.StateSpace):
    """ Implementation of DC seprated excited motor - state-space equation.

    xdot = A*[i, w, T] + B*[U, Tl]
    y = C*[i, w, T] + D*[U, Tl]
    """

    def __init__(self, R=1, J=1, L=1, mu=1, k=1, fi=1, dt=1, **opt):
        """
        Arguments
        ---------
        R : `float`
            Armature resistance
            Default ``R``=1
        J : `float`
            Rotor inertia
            Default ``J``=1
        L : `float`
            Armature inductance
            Default ``L``=1
        mu : `float`
            Rotor viscosity
            Default ``B``=1
        k : `float`
            Motor constant
            Default ``k``=1
        fi : `float`
            Magnetic flux value from excitation circuit
            Default ``fi``=1
        dt : `float`
            Sampling period.
            Default ``dt``=1
        """
        A = np.array([[-R / L, -k * fi / L],
                      [k * fi / J, -mu / J]])
        B = np.array([[1 / L, 0],
                      [0, -1 / J]])
        C = np.array([[1, 0],
                      [0, 1],
                      [k * fi, 0]])
        D = np.zeros((3, 2))

        super().__init__(ABCD=[A, B, C, D], dt=dt, name="DC motor",
                         named_output=['current', 'speed', 'torque'], **opt)

    @property
    def current(self):
        """
        Returns:
        --------
        i : `numpy.ndarray`
            Returns current vector
        """
        return self['current']

    @property
    def speed(self):
        """
        Returns:
        --------
        omega : `numpy.ndarray`
            Returns velocity vector
        """
        return self['speed']

    @property
    def torque(self):
        """
        Returns:
        --------
        Te : `numpy.ndarray`
            Returns electromagnetic moment vector
        """
        return self['torque']


"""

BLDC motor

"""


def trapezoid_func_F(th_e):
    """Calculate value of trapezoidal waveform for a given motor electrical angle.
    
    Arguments
    ---------
    th_e : `float`
        Motor electrical angle in range [0, 2pi]
        
    Returns
    -------
    F : `float`
        Returns value of trapezoidal waveform for given angle
    """
    tau = np.pi / 6
    f = 2 / 3 * np.pi
    T = np.pi
    A = 1

    if th_e <= tau:
        F = th_e / tau
    elif th_e <= f + tau:
        F = A
    elif th_e <= T:
        F = A - A / tau * (th_e - (T - tau))
    elif th_e <= T + tau:
        F = - A / tau * (th_e - T)
    elif th_e <= 2 * T - tau:
        F = -1
    elif th_e <= 2 * T:
        F = -A + A / tau * (th_e - (2 * T - tau))
    else:
        raise Exception("Could not get trapezoid value for argument th={}".format(th_e))
    return Models.convert_2dim_array(F)


def angle_state(th_m, p):
    """Calculate 3-phase back-electrical motor force
    
    Parameters
    ----------
    th_m : `float`
        Mechanical motor angle in radians
    p : `int`
        Motor number of pole pairs
    
    Returns
    -------
    F : `tuple`
        Returns back-EMF for phase A, B, C
    
    """
    th_e = p * th_m
    Fa = trapezoid_func_F(th_e % (2 * np.pi))
    Fb = trapezoid_func_F((th_e - 2 / 3 * np.pi) % (2 * np.pi))
    Fc = trapezoid_func_F((th_e - 4 / 3 * np.pi) % (2 * np.pi))
    return Fa, Fb, Fc


class BackEmf(Models.Block):
    def __init__(self, K, p, **opt):
        """
        Inputs = [w, th_m]
        Outputs = [Fa, Fb, Fc, ea, eb, ec]
        States = None

        Arguments
        ---------
        K : `float`
            Motor electric constant
        p : `int`
            Number of pole pairs
        """

        def _f(w, th, K=K, p=p, angle_f=angle_state):
            Fa, Fb, Fc = np.asanyarray(angle_f(th, p))
            ea = p * K * w * Fa
            eb = p * K * w * Fb
            ec = p * K * w * Fc

            return np.asanyarray([[Fa, Fb, Fc, ea, eb, ec]]).reshape(-1, 1)

        super().__init__(model='BLDC-Back-Emf', shape=(2, 6, 0),
                         model_func=lambda u, x, t, dt: (_f(u[0], u[1]), None), **opt)
        self.initial_state()


class Torque(Models.Block):
    def __init__(self, K, p, **opt):
        """
        Inputs = [Fa, Fb, Fc, ia, ib, ic]
        Outputs = [Te]
        States = None

        Arguments
        ---------
        K : Torque constant
        p : Number of pole pairs
        """

        def _f(F, i, K=K, p=p):
            Te = p * K * (F[0] * i[0] + F[1] * i[1] + F[2] * i[2])
            return Te.reshape(-1, 1)

        super().__init__(model='BLDC-Torque', shape=(6, 1, 0),
                         model_func=lambda u, x, t, dt: (_f(u[0:3], u[3:6]), None), **opt)
        self.initial_state()


def BLDC_ideal_commutator(i, f):
    """Calculates impulses for inverter gates.

    Parameters:
    -----------
    i : `numpy.ndarray`
        BLDC current vector.
    f : `numpy.ndarray`
        Values from BLDC ``trapezoid_func_F`` function.

    Returns:
    --------
    G : `tuple`
        Values for inverter gates.
    """
    # Gates signal
    G1 = 1 if f[0] >= 1 else 0
    G2 = 1 if f[0] <= -1 else 0
    G3 = 1 if f[1] >= 1 else 0
    G4 = 1 if f[1] <= -1 else 0
    G5 = 1 if f[2] >= 1 else 0
    G6 = 1 if f[2] <= -1 else 0

    # Diodes work
    Fd1 = 1 if i[0] < 0 and G2 == 0 else 0
    Fd2 = 1 if i[0] > 0 and G1 == 0 else 0
    Fd3 = 1 if i[1] < 0 and G4 == 0 else 0
    Fd4 = 1 if i[1] > 0 and G3 == 0 else 0
    Fd5 = 1 if i[2] < 0 and G6 == 0 else 0
    Fd6 = 1 if i[2] > 0 and G5 == 0 else 0

    # Gates final signal
    Gz1 = G1 or Fd1
    Gz2 = G2 or Fd2
    Gz3 = G3 or Fd3
    Gz4 = G4 or Fd4
    Gz5 = G5 or Fd5
    Gz6 = G6 or Fd6

    return Gz1, Gz2, Gz3, Gz4, Gz5, Gz6


def BLDC_ideal_inverter(Udc, G):
    """3-phase inverter with ideal switches and with no diodes.

    Parameters:
    -----------
    Udc : `float`
        Input voltage.
    G : `list`, `tuple`, `numpy.ndarray`
        List of gates impulses.

    Returns:
    --------
    V : `tuple`
        Returns 3-phase voltages - Va, Vb, Vc
    """
    Va = 0
    Vb = 0
    Vc = 0

    Va += Udc / 2 if G[0] else 0
    Va -= Udc / 2 if G[1] else 0

    Vb += Udc / 2 if G[2] else 0
    Vb -= Udc / 2 if G[3] else 0

    Vc += Udc / 2 if G[4] else 0
    Vc -= Udc / 2 if G[5] else 0

    return Va, Vb, Vc


class BLDC(Models.StateSpace):
    def __init__(self, R=1., L=1., J=1., mu=1., ke=1., kt=1., p=1, dt=1, **opt):
        A = [[-1 * R / L, 0, 0, 0],
             [0, -1 * R / L, 0, 0],
             [0, 0, -1 * mu / J, 0],
             [0, 0, 1, 0]]

        B = [[2 / (3 * L), 1 / (3 * L), -2 / (3 * L), -1 / (3 * L), 0, 0],
             [-1 / (3 * L), 1 / (3 * L), 1 / (3 * L), -1 / (3 * L), 0, 0],
             [0, 0, 0, 0, 1 / J, -1 / J],
             [0, 0, 0, 0, 0, 0]]

        C = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [-1, -1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        D = np.zeros((5, 6))

        super().__init__(ABCD=[A, B, C, D], dt=dt, **opt)

        self.torque = Torque(kt, p, **opt)
        self.back_Emf = BackEmf(ke, p, **opt)


    def simulate(self, inputs, dt=None):
        inputs = Models.convert_2dim_array(inputs).reshape(-1, 1)

        ea, eb, ec = self.back_Emf.now[3:6]
        uab, ubc, Tl = inputs

        U = [[uab], [ubc], [ea - eb], [eb - ec], [self.torque.now], [Tl]]
        super().simulate(inputs=U, dt=dt)
        self.back_Emf.simulate(inputs=self.now[3:5], dt=dt)  # [w, th_m]
        self.torque.simulate(inputs=[self.back_Emf.now[0:3], self.now[0:3]], dt=dt)  # [Fa, Fb, Fc, ia, ib, ic]
    pass
