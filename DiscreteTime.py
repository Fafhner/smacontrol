import numpy as np


def discretization_tustin(ABCD, T):
    """Implementation for Tustin's discretization method.
        Getting the A, B, C, D matrices and time sampling, the function performs operation
            Ad = (I + A*T/2)(I - A*T/2)^-1
            Bd = (I - A*T/2)^-1*B*sqrt(T),
            Cd = sqrt(T)*C*(I - A*T/2)^-1
            Dd = D + C*(I - A*T/2)^-1*B*T/2

        Parameters
        ----------
        ABCD : `list`, `tuple` of `numpy.ndarray'
            A holder for 2-dimensional arrays A, B, C, D. These four arrays are describing
            a state-space system.
        T : 'float'
            Time sampling of a simulation.

        Returns
        -------
        ABCDd : `tuple` of `numpy.ndarray'
            Returns four arrays of discrete time system, the equivalent of continuous time
            system discretized by Tustin's method with time sampling ``T``
    """

    ABCD = [np.asanyarray(mat) for mat in ABCD]
    A, B, C, D = [np.asanyarray([mat]) if mat.ndim == 1 else mat for mat in ABCD]
    # Checking matrices
    if A.shape[0] != B.shape[0]:
        raise Exception(
            "Invalid number of rows of A and B matrices. A.shape = {}, B.shape = {}".format(A.shape, B.shape))
    elif A.shape[1] != C.shape[1]:
        raise Exception(
            "Invalid number of columns of A and C matrices. A.shape = {}, C.shape = {}".format(A.shape, C.shape))
    elif B.shape[1] != D.shape[1]:
        raise Exception(
            "Invalid number of columns of B and D matrices. B.shape = {}, D.shape = {}".format(B.shape, D.shape))
    elif C.shape[0] != D.shape[0]:
        raise Exception(
            "Invalid number of rows of C and D matrices. C.shape = {}, D.shape = {}".format(C.shape, D.shape))

    I = np.eye(A.shape[0])  # Create diagonal I
    A_1 = np.linalg.inv((I - A * T / 2))  # (I - A*T/2)^-1
    Ad = (I + A * T / 2)@A_1
    Bd = A_1@B*T
    Cd = C@A_1
    Dd = D + 0.5*C@Bd

    return tuple((np.asanyarray(Ad), np.asanyarray(Bd), np.asanyarray(Cd), np.asanyarray(Dd)))


def calc_ss(ABCDd, x, u):
    """System simulation at time step k. 
        Calculate output and next state from discret state-space representation
            x[k+1] = Ad*x[k] + Bd*u[k]
            y[k] = Cd*x[k] + Dd*u[k]
            
        Parameters
        ----------
        ABCDd : `list`, `tuple` of `numpy.ndarray`
            A holder for discrete-time equation arrays. Must contain four
            2-dimensional arrays.
        x : `numpy.ndarray`
            A nx1 2-dim array, where n is the number of states. Contains states values at time k.
        u : `numpy.ndarray`
            A px1 2-dim array, where p is the number of inputs. Contains inputs values at time k.

        Returns
        -------
        yk : `numpy.ndarray`
            A qx1 array as an output from the state system at time k.
        xk_1 : `numpy.ndarray`
            A nx1 array as a state vector at time k+1.
    """

    Ad, Bd, Cd, Dd = [np.asarray(mat) for mat in ABCDd]

    u = np.atleast_2d([u]).reshape(-1, 1)

    xk_1 = np.dot(Ad, x).reshape(1, -1) + np.dot(Bd, u).reshape(1, -1)
    yk = np.dot(Cd, x).reshape(1, -1) + np.dot(Dd, u).reshape(1, -1)
    return yk, xk_1






