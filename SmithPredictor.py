from smacontrol import Models


class SmithPredictor(Models.Block):
    """Smith Predictor implementation
    """

    def __init__(self, reg, model, delay, filter=None, **opt):
        """
        Arguments:
        ---------
        reg : `Models.Block`
            Regulator represantation.
        model : `Models.Block`
            Model of a process.
        delay : `int`
            Dead time. Number of samples to be delayed.
        filter : `Models.Block`, optional
            Filtering block.
        opt : `dict`, optional
            `Model.Block` options.
        """
        self.controller = reg
        self.model = model
        self.model_delay = Models.Delay(d=delay, **opt)
        self.filter = filter
        super().__init__(model='Smith-Predictor', shape=(2, 1, 0), model_func=None, **opt)

    @property
    def now(self):
        return self.controller.now

    def __getitem__(self, item):
        return self.controller[item]

    def simulate(self, inputs, dt=None):
        """Simulate SP block.

        Arguments:
        ----------
        U : `numpy.ndarray`
            2-dimensional array with two rows. First row [0] contain a setting value,
            second [1] contain a return value from plant/process.
        dt : `float`
            Step time of simulation.
            Default ``dt``=1
        Returns:
        --------
        y : `numpy.ndarray`
            Last controller's value
        """

        u, yplant = inputs
        u = Models.convert_2dim_array(u)
        yplant = Models.convert_2dim_array(yplant)

        yfilter = yplant - self.model_delay.now
        if self.filter:
            yfilter = self.filter.simulate(yfilter, dt)

        ureg = u-yfilter

        yloop = Models.StateSpace.algebraic_loop_solver(ureg, self.controller, self.model, sign=-1)
        ymodel = self.model.simulate(yloop, dt)
        ycontrol = self.controller.simulate(ureg - ymodel, dt)

        self.model_delay.simulate(ymodel, dt)

        return ycontrol

