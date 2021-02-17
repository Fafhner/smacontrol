from smacontrol import Models


class IMC(Models.Memory):
    """Implementation of Internal Model Control(IMC) regulation.
    """

    def __init__(self, reg, model, delay, **opt):
        """
        Arguments:
        reg : `Models.Block`
            Controller block.
        model : `Models.Block`
            Model of a process/plant.
        opt : `dict`
            ``Models.Block`` options.
        """
        self.controller = reg
        self.model = model
        self.delay = Models.Delay(d=delay, **opt) if delay is not None else None
        super().__init__(in_len=1, output_no_mem=True, **opt)

    @property
    def now(self):
        return self.controller.now

    def __getitem__(self, item):
        return self.controller[item]

    def simulate(self, inputs, dt=1):
        """Simulate IMC.

        Arguments:
        ----------
        U : `numpy.ndarray`
            2-dimensional array with two rows. First row U[0] contain a setting value,
            second row U[1] contain a return value from plant/process.
        time : `int`
            Simulation number of iterations
            Default ``time``=1
        Returns:
        --------
        y : `numpy.ndarray`
            Last controller's value
        """

        u, yplant = inputs
        u = Models.convert_2dim_array(u)
        yplant = Models.convert_2dim_array(yplant)

        ymodel = self.delay.now if self.delay else self.model.now
        super().simulate(u + ymodel - yplant, dt=dt)

        ycontrol = self.controller.simulate(inputs=super().now, dt=dt)
        self.model.simulate(inputs=self.controller.now, dt=dt)

        if self.delay:
            self.delay.simulate(inputs=self.model.now, dt=dt)

        return ycontrol


class IMC2(Models.Memory):
    """Implementation of Internal Model Control(IMC) regulation.
    """

    def __init__(self, reg, model, **opt):
        """
        Arguments:
        reg : `Models.Block`
            Controller block.
        model : `Models.Block`
            Model of a process/plant.
        opt : `dict`
            ``Models.Block`` options.
        """
        self.controller = reg
        self.model = model

        super().__init__(in_len=1, output_no_mem=True, **opt)

    @property
    def now(self):
        return self.controller.now

    def __getitem__(self, item):
        return self.controller[item]

    def simulate(self, inputs, dt=None):
        """Simulate IMC.

        Arguments:
        ----------
        U : `numpy.ndarray`
            2-dimensional array with two rows. First row U[0] contain a setting value,
            second row U[1] contain a return value from plant/process.
        time : `int`
            Simulation number of iterations
            Default ``time``=1
        Returns:
        --------
        y : `numpy.ndarray`
            Last controller's value
        """

        u, yplant = inputs
        u = Models.convert_2dim_array(u)
        yplant = Models.convert_2dim_array(yplant)

        yloop = Models.StateSpace.algebraic_loop_solver(u - yplant, self.controller, self.model)

        ymodel = self.model.simulate(inputs=yloop, dt=dt)

        super().simulate(u + ymodel - yplant, dt=dt)

        ycontrol = self.controller.simulate(inputs=super().now, dt=dt)

        return ycontrol
