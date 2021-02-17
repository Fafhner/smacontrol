from smacontrol import DiscreteTime as dtime
import numpy as np


def convert_2dim_array(arr):
    """Convert given array to be 2-dimensional numpy array.

    Parameters
    ----------
    arr : 'list', 'tuple', 'numpy.ndarray'
        Array or nested list/tuple to be converted.

    Returns
    -------
    arr : 'numpy.ndarray'
        Converted 2-dim array.

    Raises
    ------
    Exception
        Raised if function was unable to convert given array to 2-dim numpy array.
    """

    arr = np.atleast_2d(np.squeeze(arr))

    dim = 0
    # Try merging
    while arr.ndim > 2 and dim - arr.ndim != 0:
        dim = arr.ndim
        arr = np.concatenate(arr)

    if arr.ndim != 2:
        raise Exception("Input validation: Could not convert to 2-dim array. Got {}".format(arr.ndim))
    return arr


class SimIterator:
    def __init__(self):
        self._it = 0
        self._t = 0
        self._dt = 0

    @property
    def it(self):
        return self._it

    @property
    def t(self):
        return self.t

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    def inc(self):
        self._it += 1
        self._t += self._dt


class Block:
    """Class for the purpose of modeling control systems and simulation

        Basic class, mostly used for inheritance.

        Attributes
        ----------
        _model : `str`
            Name of the class. If class is inherited, then the `_model` name
            should be the child class's name.
        _model_func : 'function'
            Function object, that simulate the behaviour of the model.
            The function header should accept arguments as foo('u', 'x', 't'), where
            'u' is an input vector at time 't'', 'x' is a state vector at time 't'
            and 't' is given time iterator ``_it``. The function should return a
            tuple of values '(yk, xk_1)', where 'yk' is output at time 't',
            'xk_1' is next state at time 't+1'. If object does not contain
            any states, then function should return '(yk, None)'.
        _ioxshape : `tuple`
            Shape of the block. Must contains three values : number of inputs 'i',
            number of outputs 'o', number of states 'x' in a form ('i', 'o', 'x')
        _named_state : `dict` of `(int, str)`
            Dictionary. Key is the Block state's number and the value is a name
            of that state.
        _named_output : `dict` of `(int, str)`
            Dictionary. Key is the Block input's number and the value is a name
            of that input.
        Y0 = : `numpy.ndarray`
            Numpy.ndarray qx1 array, where q is the number of inputs. ``Y0`` is
            an initial output vector. When simulation at time k=0 needs a
            previous value of this object, it returns this vector.
        _state : `numpy.ndarray`
            2 dimensional array for accumulating states' values. The number
            of rows(states) is specified by ``_ioxshape``.
        _output : `numpy.ndarray`
            2 dimensional array for accumulating inputs' values. The number
            of rows(inputs) is specified by ``_ioxshape``.
        _it : `int`
            Inner ssimulation iterator. Incremented for every call of ``simulate``
            method.
        len : `int`
            Number of columns of ``_state`` and ``_output`` columns.
            If number of samples exceed the ``_state`` or ``_output`` length,
            then ``len``, takes part in ``resize`` method
        resize_k : `float`
            Resize gain used along with ``len``, when ``_state`` and ``_output``
            arrays overflow.
        state_no_mem : `bool`
            If true, then ``_state`` array has only one column and keep only vector
            of values at given time. Otherwise all samples are preserved. Do nothing
            if object has no states.
        output_no_mem : `bool`
            Same as ``state_no_mem``, but for ``_output`` vector.
        name : `str`
            Name of the object.

        Notes
        -----
        The sole difference between ``_delay`` and ``_in_delay`` is for flexibility.
            If the system has  blocks of type SISO, then ``_in_delay`` can be used when only
            input of one of the blocks must be delayed. With ``_delay`` it delayes
            all connected inputs to output, creating the situation where there is only
            one delay created, than making delay at every input.

    """

    def __init__(self, shape, model_func, model='Block', **opt):
        """
            Arguments
            ---------
            model : 'str'
                Model name
            _model_func : 'function'
                Function object, that simulate the behaviour of the model.
                The function header should accept arguments as foo('u', 'x', 't'), where
                'u' is an input vector at time 't'', 'x' is a state vector at time 't'
                and 't' is given time iterator ``_it``. The function should return a
                tuple of values '(yk, xk_1)', where 'yk' is output at time 't',
                'xk_1' is next state at time 't+1'. If object does not contain
                any states, then function should return '(yk, None)'.
            shape : 'tuple'
                the shape of a model. Must be a tuple as (i, o, x), where 'i' is the number of inputs, 'o' the
                number of outputs, 'x' the number of states. Must be greater or equal 0.
            **opt:
                Avaiable options:
                -dt  - System sampling time. 
                            Default ``dt``=1
                -state_no_mem - If False saves all state values from previous iteration.
                            Otherwise, only current values are preserved.
                            Default ``state_no_mem``=True
                -output_no_mem - If False saves all output values from previous iteration.
                            Otherwise, only current values are preserved.
                            Default 'output_no_mem'=False
                -len - specifies initial size of output and state arrays.
                            It is prefered, to initialize 'len' argument with full simulation time
                            in order not to make unnecessary copies.
                            Default 'len'=1000
                -resize_k - specifies resize factor of an output/state arrays.
                            If iteration time overstep a length of the arrays, then
                            the arrays increase their sizes by the value:
                                new_len:=resize_k * actual_len
                            for 'resize_k' > 0. If 'resize_k' = 0, then actual length is incremented
                            by initial 'len' optional argument - new_len:=len + actual_len.
                            Default: 'resize_k' = 0
                -name - specifies the object name.
                            Default: 'name'=this
                -named_state - specifies name for each state, to use as an index.
                            Can be in a form of tuple/list or dict, where key is the state number and
                            the value is the state name.
                            Default: 'named_state'=None
                -named_output - specifies name for each output, to use as an index.
                            Can be in a form of tuple/list or dict, where key is the output number and
                            the value is the output name.
                            Default: 'named_output'=None
                -array_y - insert an array as output array.
                            The numbers of rows must be the same as specified in ``ioxshape``
                -array_x - insert an array as state array.
                            The numbers of rows must be the same as specified in ``ioxshape``
        """

        """Initializing """
        self._model = model
        self._model_func = model_func
        self._ioxshape = tuple(shape)
        self._named_state = None
        self._named_output = None
        self.Y0 = np.zeros((self._ioxshape[1], 1))
        self.X0 = np.zeros((self._ioxshape[2], 1)) if self._ioxshape[2] != 0 else None
        self._state = None
        self._output = None
        self._it = 0

        self.dt = opt['dt'] if 'dt' in opt else 1
        self.v_len = opt['len'] if 'len' in opt else 1000
        self.resize_k = opt['resize_k'] if 'resize_k' in opt else 0
        self.state_no_mem = opt['state_no_mem'] if 'state_no_mem' in opt else True
        self.output_no_mem = opt['output_no_mem'] if 'output_no_mem' in opt else False
        self.name = opt['name'] if 'name' in opt else str(self)

        """Creating state vectors"""
        if self.ioxshape[2] != 0:
            x_cols = self.v_len + 1 if not self.state_no_mem else 1
            x_rows = self.ioxshape[2]
            arr_x = None if 'array_x' not in opt else opt['array_x']
            if arr_x is not None:
                arr_x = convert_2dim_array(arr_x)
                if arr_x.shape[0] != self.ioxshape[1]:
                    raise Exception("Initlizing array: Number of rows of given y-array is not equal to the number of" \
                                    "outputs.Got {}, expected {}".format(arr_x.shape[0], self.ioxshape[2]))

            self._state = np.zeros((x_rows, x_cols)) if arr_x is None else arr_x

        """Creating output vectors """
        y_cols = self.v_len if not self.output_no_mem else 1
        y_rows = self.ioxshape[1]
        arr_y = None if 'array_y' not in opt else opt['array_y']
        if arr_y is not None:
            arr_y = convert_2dim_array(arr_y)
            if arr_y.shape[0] != self.ioxshape[1]:
                raise Exception("Initlizing array: Number of rows of given y-array is not equal to the number of" \
                                "outputs.Got {}, expected {}".format(arr_y.shape[0], self.ioxshape[1]))
        self._output = np.zeros((y_rows, y_cols)) if arr_y is None else arr_y

        """Creating name"""
        if 'named_state' in opt:
            self.setup_named_states(opt['named_state'])

        if 'named_output' in opt:
            self.setup_named_outputs(opt['named_output'])

    pass

    """Class methods"""

    @property
    def ioxshape(self):
        return self._ioxshape

    @property
    def now(self):
        """Property

        Returns
        -------
        output : `numpy.ndarray`
            Return an output array at time `self.it`-1 or initial output, when `self.it`-1 < 0.
        """
        if not self.output_no_mem:
            val = self.__getitem__(index=(slice(None, None, None), self._it - 1)) if self._it > 0 else self.Y0
        else:
            val = self._output[:, 0]

        return convert_2dim_array(val).reshape(-1, 1)

    @property
    def state_now(self):
        x = self._state[:, (self._it - 1) % self._state.shape[1]] if self._it - 1 >= 0 else self.X0
        return convert_2dim_array(x).reshape(-1, 1)

    @property
    def state_it(self):
        return self._it % self._state.shape[1]

    @property
    def output_it(self):
        return self._it % self._output.shape[1]

    def get_by_name(self, name):
        """Return an output/state array by their names

        If ``_output`` and ``_state`` have the same name for output/state
        then it returns array for output only.

        Arguments
        ---------
        name : 'str'
            Name of one of the ``_output`` or ``_states`` vector.

        Returns
        -------
            arr : `numpy.ndarray`
                ``_output`` or ``_states`` vector with length of <0, ``self._it``)
        """
        if name in self._named_output:
            return self._output[self._named_output[name], 0:self._it]
        elif self._named_state is not None and name in self._named_state:
            return self._state[self._named_state[name], 0:self._it]
        else:
            raise Exception("{} not found in named state nor named output.".format(name))

    def setup_shape(self, ioxshape):
        if len(ioxshape) != 3:
            raise Exception("Setup shape: Wrong size of shape. Excpected 3 elements, got {}".format(len(ioxshape)))

        if not all([isinstance(_, int) for _ in ioxshape]):
            raise Exception("Setup shape: Not all elements consist a type of int")

        self._ioxshape = tuple(ioxshape)
        self._output = np.zeros((self.ioxshape[1], self.v_len))
        self._state = np.zeros((self.ioxshape[2], self.v_len)) if self.ioxshape[2] != 0 else None

    def setup_named_states(self, x_name):
        if self._state is None:
            raise Exception("Block does not have any state")
        if isinstance(x_name, (list, tuple)):
            if len(x_name) != self.ioxshape[2]:
                err = "state names option: Incorrect length of state names list. Expected {}, got {}".format(
                    self.ioxshape[2], x_name)
                raise Exception(err)
            x_name = dict(((x_name[i], i) for i in range(0, len(x_name))))
        self._named_state = x_name

    def setup_named_outputs(self, y_name):
        if isinstance(y_name, (list, tuple)):
            if len(y_name) != self.ioxshape[1]:
                err = "Output names option: Incorrect length of output names list. Expected {}, got {}".format(
                    self.ioxshape[1], y_name)
                raise Exception(err)
            y_name = dict(((y_name[i], i) for i in range(0, len(y_name))))
        self._named_output = y_name

    def initial_state(self, X0=None):
        """Calculate initial output at state ``X0``

        The output is calculated with the states ``X0`` and with zero array as the input.
        ``_it`` is not incremented in this function.

        Arguments
        ---------
        X0 : `numpy.ndarray`
            Array of states, with size nx1, where 'n' is the number of states.
            If ``X0``=None, then vector filled with zeros is created.
            Default ``X0``=None
        """
        u = convert_2dim_array(np.zeros((self.ioxshape[0], 1))).reshape(-1, 1)

        if X0 is not None:
            self.X0 = convert_2dim_array(X0)
        elif self.ioxshape[2] != 0:
            self.X0 = np.zeros((self.ioxshape[2], 1))

        yk, xk_1 = self._model_func(u, self.X0, 0, self.dt)
        yk = convert_2dim_array(yk).reshape(-1, 1)

        self.Y0 = yk
        if self._state is not None:
            xk_1 = convert_2dim_array(xk_1).reshape(-1, 1)
            self._state[:, 0] = xk_1[:, 0]

    def cut(self, save=False):
        """Resize ``_output`` and ``_state`` vector to minimal size.

        Shorten the output and the state arrays, removing unnecesary zeros. The final
        length of arrays is equal to 'self.it' attribute.

        Arguments
        ---------
        save : `bool`
            If 'save'=False, will also remove the next state from state array. Otherwise the last state is preserved and
            is possible to continue the simulation
            Default ``save``=False
        """
        it = self._it
        self._output = self._output[:, 0:it]

        if self._state is not None:
            if save:  it += 1
            self._state = self._state[:, 0:it]

    def resize(self):
        """Resize output and state vector accordingly to the ``self.len`` and ``self.resize_k`` values.
        """
        # Check for state vector
        if not self.state_no_mem:
            if self._it >= self._state.shape[1]:
                add_len = self.v_len if self.resize_k == 0 else int(self.resize_k * self._state.shape[1])
                self._state = np.concatenate((self._state, np.zeros((self._state.shape[0], add_len))), axis=1)

        # Check for output vector
        if not self.output_no_mem:
            if self._it >= self._output.shape[1]:
                if self._it >= self._output.shape[1]:
                    add_len = self.v_len if self.resize_k == 0 else int(self.resize_k * self._output.shape[1])
                    self._output = np.concatenate((self._output, np.zeros((self._output.shape[0], add_len))), axis=1)

    def clear(self):
        if self._state is not None:
            self._state = np.zeros((self.ioxshape[2], self.v_len + 1)) if not self.state_no_mem else np.zeros(
                (self.ioxshape[2], 1))

        self._output = np.zeros((self.ioxshape[1], self.v_len + 1)) if not self.output_no_mem else np.zeros(
            (self.ioxshape[1], 1))

        self._it = 0

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get_by_name(index)
        elif isinstance(index, int):
            if self.output_no_mem:
                return self._output[0, :]
            return self._output[0, index]

        if isinstance(index, tuple):
            r = index[0]
            c = index[1]
        else:  # index = slice
            r = 0
            c = index

            if self.ioxshape[1] > 1 and r == 0:
                print(
                    "Block warning:{}->{} has more than one output. Returning only one at index (0, {}).".format(
                        self._model,
                        self.name,
                        c))
        return self._output[r, c]

    def simulate(self, inputs, dt=None):
        """Simulate the output of the block.

        Arguments:
        ----------
        inputs : `numpy.ndarray`
            Input array. If block has many inputs and ``inputs`` argument is a 1-dim vector
            or 2-dim 1xp array with one row and ``time``=1, then ``inputs`` is converted as
            2-dim px1 array, with 'p' as number of object inputs.
        dt : `float`, optional
            If not None, then is forwarded to ``self._model_func``. Otherwise the ``self.dt`` class 
            attribute is used.
            Default ``dt``=None
        """
        inputs = convert_2dim_array(inputs).reshape(-1, 1)

        if dt is None:
            dt = self.dt

        self.resize()
        # Input and state acquisition
        u = inputs.copy()
        x = self._state[:, self.state_it] if self._state is not None else None

        # Calculate output and state
        yk, xk_1 = self._model_func(u, x, self._it, dt)

        # Process output
        yk = convert_2dim_array(yk).reshape(-1, 1)

        # Save result
        self._output[:, self.output_it] = yk[:, 0]

        self._it += 1

        # Save states
        if self._state is not None:
            xk_1 = convert_2dim_array(xk_1).reshape(-1, 1) if xk_1 is not None else None
            self._state[:, self.state_it] = xk_1[:, 0]

        return yk


class Derivative(Block):
    """Derivative block.
    """

    def __init__(self, Kd, dt, **opt):
        """Function y[n] = K*(u[n]-u[n-1])/dt

        Arguments:
        ----------
        Kd : `float`
            Derivative gain factor.
        dt : `float`
            Sampling period.
        opt : `dict`
            ``Models.Block`` options.
        """

        def f(u, x, t, dt, K=Kd):
            # x = u[n-1]
            return K / dt * (u - x), u  # y, x

        shape = (1, 1, 1)
        super().__init__(model='derivative', shape=shape, model_func=f, dt=dt, **opt)


class Integral(Block):
    """Integral block.
    """

    def __init__(self, Ki, dt, type='forward', **opt):
        """Function y[n] = y[n-1] + Ki * dt * u[n].

        Arguments:
        ----------
        Ki : `float`
            Integral gain factor.
        dt : `float`
            Sampling period.
        type : `string`
            Type of integration - trapeziodal
        opt : `dict`
            ``Models.Block`` options.
        """

        shape = (1, 1, 2)
        if type == 'backward':
            shape = (1, 1, 1)


        super().__init__(model='integral', shape=shape, model_func=None, dt=dt, **opt)

        # Change type of integration
        self.change_type(Ki, type)

    @staticmethod
    def backward(u, x, t, dt, K):
        """y[n] = y[n-1] + K*dt*u[n]
        x[0] = y[n-1]
        """
        y = x[0] + K * dt * u
        return y, y

    @staticmethod
    def forward(u, x, t, dt, K):
        """y[n] = y[n-1] + K*dt*u[n-1]
        x[0] = y[n-1]
        x[1] = u[n-1]
        """
        y = x[0] + K * dt * x[1]
        return y, [y, u]

    @staticmethod
    def trapezoidal(u, x, t, dt, K):
        """y[n] = y[n-1] + K*dt*(u[n] - u[n-1])/2
        x[0] = y[n-1]
        x[1] = u[n-1]
        """
        y = x[0] + K * dt * (u + x[1]) / 2
        return y, [y, u]

    def change_type(self, Ki, type):
        """Change type of integration

        Arguments:
        ----------
        Ki = `float`
            Integration gain.
        type : `str`
            Name of integration. Possible: `forward`, `backward`, `trapezoidal`.
        """

        shape = (1, 1, 2)
        if type == 'backward':
            shape = (1, 1, 1)

        integral = getattr(Integral, type)

        def f(u, x, t, dt, K=Ki):
            return integral(u, x, t, dt, K)

        self.setup_shape(shape)
        self._model_func = f




class Gain(Block):
    """Gain block.
    """

    def __init__(self, Kp, **opt):
        """
        Arguments:
        ----------
        Kp : `float`
            Gain coefficient, y=``Kp``*u.
        opt : `dict`, optional
            ``Models.Block`` options.
        """

        def f(u, x, t, dt, K=Kp):
            return K * u, None  # y, x

        shape = (1, 1, 0)
        super().__init__(model='gain', shape=shape, model_func=f, **opt)


class Memory(Block):
    """Storage block, for keeping inputs.
    """

    def __init__(self, in_len, **opt):
        """
        Arguments:
        ----------
        in_len : `int`
            Number of inputs.
        opt : `dict`
            ``Models.Block`` options.
        """

        def f(u, x, t, dt):
            return u, None  # y, x

        super().__init__(model='Memory', shape=(in_len, in_len, 0), model_func=f, **opt)


class StateSpace(Block):
    """Discrete state-space represantation of a block from continuous matrices A, B, C, D.
        This class uses Tustin's method for discretization.
    """

    def __init__(self, ABCD, dt=1, K=1, discrete=False, **opt):
        """
        Arguments:
        ----------
        ABCD : `list`, `tuple`
            List of continuous matrices A, B, C, D of a state-space model.
        dt : `float`, optional
            Sampling period.
            Default ``dt``=1
        K : `float`, optional
            Output gain factor y' = K*y.
        discrete : `bool`, optional
            If True, then arrays ``ABCD`` are discrete and the discretization is not performed.
            Default ``discrete``=Dalse
        opt : `dict`
            ``Models.Block`` options.
        """
        if len(ABCD) != 4:
            err = "Model representation: Expected 4 elements, got {}.".format(len(ABCD))
            raise Exception(err)
        ABCDd = dtime.discretization_tustin([np.asanyarray(m) for m in ABCD], T=dt) if not discrete else ABCD

        self._disc_m = ABCDd

        def f(u, x, t, dt, mat=self._disc_m, gain=K):
            yk, xk_1 = dtime.calc_ss(mat, x, u)
            return yk * gain, xk_1

        shape = (ABCDd[1].shape[1], ABCDd[2].shape[0], ABCDd[0].shape[0])
        super().__init__(model='state-space', shape=shape, model_func=f, dt=dt, **opt)

    @property
    def mat(self):
        return self._disc_m

    @staticmethod
    def algebraic_loop_solver(u, obj1, obj2, sign=1):
        """Only for SISO StateSpace class objects.

        y = [C1*x1 + D1*u +/- D1*C2*x2]*[I -/+ D1*D2]^-1
        """
        C1, D1, x1 = obj1.mat[2], obj1.mat[3], obj1.state_now
        C2, D2, x2 = obj2.mat[2], obj2.mat[3], obj2.state_now

        I = np.eye(D1.shape[0])
        y = (np.dot(C1, x1) + np.dot(D1, u) + sign * np.dot(np.dot(D1, C2), x2)) * np.linalg.inv(
            (I - sign * np.dot(D1, D2)))

        return convert_2dim_array(y)


class Delay(Block):
    """Block for Simulation of dead time.
    """
    def __init__(self, d, **opt):
        """
        Arguments:
        d : `int`
            Numbers of samples to be delayed.
        **opt : `dict`, optional
            Optional parameters for ``Block`` class contructor.
        """
        self._itd = -d
        self._ddelay = np.zeros((1, d))

        def f(u, x, t, dt):
            return u, None

        super().__init__(shape=(1, 1, 0), model_func=f, **opt)

    @property
    def now(self):
        return self._ddelay[:, self._itd % self._ddelay.shape[1]] if self._itd >= 0 else np.zeros((1, 1))

    @property
    def prev(self):
        return self._ddelay[:, (self._itd - 1) % self._ddelay.shape[1]] if self._itd > 0 else np.zeros((1, 1))

    def simulate(self, inputs, dt=None):
        inputs = convert_2dim_array(inputs)
        y = self._ddelay[:, self._it % self._ddelay.shape[1]].copy()
        self._ddelay[:, self._it % self._ddelay.shape[1]] = inputs.copy()

        super().simulate(inputs=y, dt=dt)
        self._itd += 1

        return self.now
