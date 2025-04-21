# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import numpy as np
import control

from . import base


class LinearStateSpace(base.RealtimeSystem):
    """
    A class for representing a linear state-space system in real-time applications.

    The LinearStateSpace class extends the RealtimeSystem class by adding functionality for handling
    linear state-space models, either given directly or in matrix form.

    Attributes:
        ss_model (control.StateSpace or None): The state-space model of the system.
        ss_matrices (tuple or None): A tuple containing the (A, B, C, D) matrices of the state-space model.
        continuous (bool): Indicates whether the state-space model is continuous or discrete.
        initial_conditions (np.ndarray or None): The initial conditions for the state variables.
        state_size (int or None): The number of state variables in the system.
    """

    def __init__(self, inputs, outputs, name=None,
                 ss_model=None, ss_matrices=None, continuous=True, initial_conditions=None, **kwargs):
        """
        Initializes the LinearStateSpace system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            ss_model (control.StateSpace, optional): The state-space model of the system. Defaults to None.
            ss_matrices (tuple, optional): A tuple containing the (A, B, C, D) matrices of the state-space model. Defaults to None.
            continuous (bool, optional): Indicates whether the state-space model is continuous or discrete. Defaults to True.
            initial_conditions (np.ndarray, optional): The initial conditions for the state variables. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, **kwargs)
        self.ss_model = ss_model
        self.ss_matrices = ss_matrices
        self.continuous = continuous
        self.initial_conditions = initial_conditions
        self.state_size = None

    def compile_system(self):
        if self.ss_model is not None:
            pass
        elif self.ss_matrices is not None:
            self.ss_model = control.ss(*self.ss_matrices)
        else:
            raise ValueError('Either ss_model or ss_matrices must not be none.')
        if self.continuous:
            self.ss_model = self.ss_model.sample(Ts=self.time_step)
        self.state_size = self.ss_model.nstates
        self.initial_conditions = base.set_default_array(value=self.initial_conditions, default=np.zeros(self.state_size))
        if np.any(self.ss_model.D != 0):
            self.has_feedthrough = True

    def get_state(self):
        return self.initial_conditions

    def update_state(self, state, input_):
        return self.ss_model.A @ state + self.ss_model.B @ input_

    def update_output(self, state, input_):
        if self.has_feedthrough:
            return self.ss_model.C @ state + self.ss_model.D @ input_
        else:
            return self.ss_model.C @ state


class TransferFunction(LinearStateSpace):
    """
    A class for representing a transfer function system in real-time applications.

    The TransferFunction class extends the LinearStateSpace class by adding functionality for handling
    transfer function models, either given directly or in matrix form.

    Attributes:
        tf_model (control.TransferFunction or None): The transfer function model of the system.
        tf_matrices (tuple or None): A tuple containing the numerator and denominator of the transfer function.
    """

    def __init__(self, inputs, outputs, name=None, tf_model=None, tf_matrices=None, **kwargs):
        """
        Initializes the TransferFunction system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str or None): The input variables for the system.
            outputs (list or str or None): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            tf_model (control.TransferFunction, optional): The transfer function model of the system. Defaults to None.
            tf_matrices (tuple, optional): A tuple containing the numerator and denominator of the transfer function. Defaults to None.
            **kwargs: Additional keyword arguments passed to the LinearStateSpace initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, **kwargs)
        self.tf_model = tf_model
        self.tf_matrices = tf_matrices

    def compile_system(self):
        if self.tf_model is not None:
            pass
        elif self.tf_matrices is not None:
            self.tf_model = control.tf(*self.tf_matrices)
        else:
            raise ValueError('Either tf_model or tf_matrices must not be none.')
        self.ss_model = control.ss(self.tf_model)
        super().compile_system()
        # TODO: do something with initial_conditions


class PIDController(base.RealtimeSystem):
    """
    A class for implementing a PID (Proportional-Integral-Derivative) controller in real-time applications.

    The PIDController class extends the RealtimeSystem class by adding functionality for controlling a system
    using a PID control algorithm.

    Attributes:
        maximum (float): The maximum output value of the controller.
        minimum (float): The minimum output value of the controller.
        rate (float): The maximum rate of change of the output value.
        offset (float): The output offset value.
        pid (control.StateSpace): The state-space representation of the PID controller.
        windup_offset (float): The offset for anti-windup correction.
        estimated_output (float): The estimated output value of the controller.
        state_size (int or None): The number of state variables in the controller.
        initial_conditions (np.ndarray or None): The initial conditions for the state variables.
    """

    def __init__(self, inputs, outputs, kp, ki, kd, tau,
                 name=None, maximum=np.inf, minimum=-np.inf, rate=np.inf, offset=0, **kwargs):
        """
        Initializes the PIDController system with the given inputs, outputs, and PID parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            kp (float): The proportional gain of the PID controller.
            ki (float): The integral gain of the PID controller.
            kd (float): The derivative gain of the PID controller.
            tau (float): The time constant for the derivative term.
            name (str, optional): The name of the system. Defaults to None.
            maximum (float, optional): The maximum output value of the controller. Defaults to np.inf.
            minimum (float, optional): The minimum output value of the controller. Defaults to -np.inf.
            rate (float, optional): The maximum rate of change of the output value. Defaults to np.inf.
            offset (float, optional): The output offset value. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, **kwargs)
        self.maximum = maximum
        self.minimum = minimum
        self.rate = rate
        self.offset = offset
        A = [[0, 0],
             [0, -1 / tau]]
        B = [[1],
             [1]]
        C = [[ki, -kd / tau ** 2]]
        D = [[kd / tau + kp]]
        self.pid = control.ss(A, B, C, D)
        self.windup_offset = 0.
        self.has_feedthrough = True
        self.estimated_output = self.offset
        self.state_size = None
        self.initial_conditions = None

    def compile_system(self):
        self.pid = self.pid.sample(Ts=self.time_step)
        self.state_size = self.pid.nstates
        self.initial_conditions = np.zeros(self.state_size)

    def get_state(self):
        return self.initial_conditions

    def update_state(self, state, input_):
        return self.pid.A @ state + self.pid.B @ input_

    def update_output(self, state, input_):
        calculated_output = self.pid.C @ state + self.pid.D @ input_ + self.offset + self.windup_offset

        if calculated_output >= self.maximum:
            temp_output = self.maximum
        elif calculated_output <= self.minimum:
            temp_output = self.minimum
        else:
            temp_output = calculated_output

        if abs(temp_output - self.estimated_output) / self.time_step > self.rate:
            temp_output = self.estimated_output + np.sign(
                temp_output - self.estimated_output) * self.rate * self.time_step

        self.windup_offset += temp_output - calculated_output
        self.estimated_output = temp_output
        return temp_output


class LowPassFilter(base.RealtimeSystem):
    """
    A class for implementing a low-pass filter in real-time applications.

    The LowPassFilter class extends the RealtimeSystem class by adding functionality for filtering input signals
    to remove high-frequency components.

    Attributes:
        cutoff_frequency (float): The cutoff frequency of the filter in Hz.
        sample_frequency (float): The sampling frequency in Hz.
        initial_condition (float): The initial output value of the filter.
        dt (float): The time step between samples.
        RC (float): The RC time constant of the filter.
        alpha (float): The filter coefficient.
        previous_output (float): The previous output value of the filter.
    """

    def __init__(self, inputs, outputs, cutoff_frequency, sample_frequency, initial_condition, name=None, **kwargs):
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        """
        Initializes the LowPassFilter system with the given inputs, outputs, and filter parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            cutoff_frequency (float): The cutoff frequency of the filter in Hz.
            sample_frequency (float): The sampling frequency in Hz.
            initial_condition (float): The initial output value of the filter.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        self.cutoff_frequency = cutoff_frequency
        self.sample_frequency = sample_frequency
        self.initial_condition = initial_condition

        # Calculate the RC time constant and alpha
        self.dt = 1 / self.sample_frequency
        self.RC = 1 / (2 * np.pi * self.cutoff_frequency)
        self.alpha = self.dt / (self.dt + self.RC)

        # self.T = 1 / self.sample_rate
        # self.omega_c = 2 * np.pi * self.cutoff_frequency
        # self.alpha = 2 / (self.T * self.omega_c + 1)

        # Initialize the previous output to the initial condition
        self.previous_output = self.initial_condition

    def update_output(self, state, input_):
        """
        Process a single input value through the filter.

        Parameters:
        - input_value: The new input value.

        Returns:
        - The filtered output value.
        """
        output_value = self.alpha * input_ + (1 - self.alpha) * self.previous_output
        self.previous_output = output_value
        return output_value


class SmoothRandomWalk(TransferFunction):
    """
    A class for generating a smooth random walk signal in real-time applications.

    The SmoothRandomWalk class extends the TransferFunction class by adding functionality for generating
    a smooth random walk signal with specified time constant and update frequency.

    Attributes:
        time_constant (float): The time constant of the random walk.
        update_frequency (int): The frequency at which the random walk is updated.
        iteration (int): The current iteration count.
        random_input (np.ndarray or None): The current random input value.
    """

    def __init__(self, outputs, name=None, time_constant=1, update_frequency=1, **kwargs):
        """
        Initializes the SmoothRandomWalk system with the given outputs, time constant, and update frequency.

        Args:
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            time_constant (float, optional): The time constant of the random walk. Defaults to 1.
            update_frequency (int, optional): The frequency at which the random walk is updated. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the TransferFunction initializer.
        """
        super().__init__(name=name, inputs=None, outputs=outputs, **kwargs)
        self.time_constant = time_constant
        self.update_frequency = update_frequency
        self.iteration = 0
        self.random_input = None

    def compile_system(self):
        base_num = [self.time_constant]
        base_den = [1, self.time_constant]
        if self.output_size == 1:
            num = base_num
            den = base_den
        else:
            num = [[[0] for _ in range(self.output_size)] for _ in range(self.output_size)]
            den = [[[1] for _ in range(self.output_size)] for _ in range(self.output_size)]
            for i in range(self.output_size):
                num[i][i] = base_num
                den[i][i] = base_den
        self.tf_matrices = [num, den]
        super().compile_system()

    def update_state(self, state, input_):
        if self.iteration == 0:
            self.random_input = np.random.normal(size=(self.output_size,))
        self.iteration = (self.iteration + 1) % self.update_frequency
        return super().update_state(state, self.random_input)

    def update_output(self, state, input_):
        return super().update_output(state, self.random_input)


class FirstOrderDiff(base.RealtimeSystem):
    """
    A class for implementing a first-order differentiator in real-time systems.

    The FirstOrderDiff class extends the RealtimeSystem class by adding functionality for computing the difference
    between the current input and the previous state, scaled by a gain factor.

    Attributes:
        gain (float): The gain factor applied to the difference.
        state (float): The previous state of the system.
    """
    def __init__(self, inputs, outputs, gain=1., initial_condition=0, name=None, **kwargs):
        """
        Initializes the FirstOrderDiff system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            gain (float, optional): The gain factor applied to the difference. Defaults to 1.0.
            initial_condition (float, optional): The initial state of the system. Defaults to 0.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.gain = gain
        self.state = initial_condition

    def update_output(self, state, input_):
        """
        Computes the difference between the current input and the previous state, scaled by the gain factor.

        Args:
            state (float): The current state of the system.
            input_ (float): The current input to the system.

        Returns:
            float: The scaled difference between the current input and the previous state.
        """
        difference = self.gain * (input_ - self.state)
        self.state = input_
        return difference
