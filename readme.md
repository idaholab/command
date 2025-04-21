Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

# Project Setup Guide

These instructions assume that you have downloaded the command repository and have a separate working folder for any input scripts and other associated files.

## Setting Up the Virtual Environment

1. In a terminal window, navigate into the command directory with the `requirements.txt` file.

1. Create a new virtual environment and install the packages from the `requirements.txt` file:
    ```
    conda create --name <env_name> --file requirements.txt
    ```

1. Activate the new environment:
    ```
    conda activate <env_name>
    ```

## Setting Up InfluxDB

1. Download InfluxDB OSS from [Influx Downloads](https://portal.influxdata.com/downloads/).

1. Start InfluxDB from the command line:
    ```
    influxd
    ```

1. Open a web browser and navigate to [http://localhost:8086](http://localhost:8086) (does not require internet).

1. Create a username and password, and follow the prompts.

1. During setup, it will provide a token for HTTP requests. Copy this token and paste it in a file called `.env` that must be located in the working directory. The `.env` file should look similar to the following:
    ```
    bucket='<insert bucket name used for influx setup>'
    url='http://localhost:8086'
    org='<insert org used for influx setup>'
    token='<insert HTTP token provided at influx setup>'
    ```

## Starting the Test Code

1. The command repository includes a test file called `main_test.py`. To run this file, make a copy of the file and place it in your working directory.

1. Adjust the sys.path.append() function (located at the top of the test file) as needed to ensure that the command package is included in the file path.

1. Run the test code using:
    ```
    python main_test.py
    ```
   
This script can then be used as a template for creating new input scripts and running more complicated simulations. 

## Technical Details

For a detailed report on the command building blocks and their interconnections, refer to [INLRPT-23-75289: Creating a Simulation Platform for Research and Development of Advanced Control Methods](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_72851.pdf).


## Creating Custom RealTimeSystems

In the command package, the two primary parent classes are the variable and system classes:
- **Variables** are the main information building blocks representing the data that can be passed back and forth between systems.
- **Systems** are the main functional building blocks that perform transformations using information.

Nearly all systems that you will create inherit from the `RealTimeSystem` class. Each `RealTimeSystem` has a time step and performs a set of functions every instance of that time step. There are `RealTimeSystem` classes already defined in various command modules that can be used in simulations. However, much of the power of the command package comes from the ability to create custom `RealTimeSystem` classes and use them in the simulation framework. 

All custom `RealTimeSystem` classes inherit from this class:

```python
class RealtimeSystem(System):
    """
    A base class for real-time systems used in simulations.

    The RealtimeSystem class extends the System class by adding real-time capabilities and methods specific to real-time simulation. It provides a framework for initializing, updating state, and managing inputs and outputs for real-time systems.

    Attributes:
        is_real_time (bool): Indicates if the system runs in real-time.
        has_feedthrough (bool): Indicates if the system has feedthrough, meaning the outputs depend directly on the inputs.
    """

    def __init__(self, name=None, inputs=None, outputs=None, time_step=None, has_feedthrough=False):
        """
        Initializes the RealtimeSystem with optional name, inputs, outputs, time step, and feedthrough flag.

        Args:
            name (str, optional): The name of the system. Defaults to None.
            inputs (list or str or None, optional): The input variables for the system. Defaults to None.
            outputs (list or str or None, optional): The output variables for the system. Defaults to None.
            time_step (float or None, optional): The time step for the system. Defaults to None.
            has_feedthrough (bool, optional): Indicates if the system has feedthrough. Defaults to False.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, time_step=time_step)
        self.is_real_time = True
        self.has_feedthrough = has_feedthrough

    def initialize(self):
        """
        Placeholder method for initializing the system. Should be overridden by subclasses.
        """
        pass

    def get_state(self):
        """
        Gets the current state of the system.

        Returns:
            np.ndarray: An array representing the current state of the system.
        """
        return np.array([])

    def update_state(self, state, input_):
        """
        Updates the state of the system based on the current state and input.

        Args:
            state (np.ndarray): The current state of the system.
            input_ (np.ndarray or None): The input to the system.

        Returns:
            np.ndarray: The updated state of the system.
        """
        return np.array([])

    def update_output(self, state, input_):
        """
        Updates the output of the system based on the current state and input.

        Args:
            state (np.ndarray): The current state of the system.
            input_ (np.ndarray or None): The input to the system.

        Returns:
            np.ndarray: The updated output of the system.
        """
        return np.array([])

    def pull_input_steps(self):
        """
        Waits for the input variables to be ready and then pulls the input data.

        Returns:
            np.ndarray: An array representing the input to the system.
        """
        self.event_manager.wait_variables_done(self.inputs)
        input_ = self.debug_wrapper(self.data_manager.pull, 'data_manager.pull', self.inputs)
        self.event_manager.set_pull_done(self)
        return input_

    def update_and_push_output(self, state, input_):
        """
        Updates the output of the system and pushes it to the data manager.

        Args:
            state (np.ndarray): The current state of the system.
            input_ (np.ndarray or None): The input to the system.
        """
        output = self.debug_wrapper(self.update_output, 'update_output', state, input_)
        self.debug_wrapper(self.data_manager.push, 'data_manager.push', output, self.outputs)
        self.event_manager.set_variables_done(self.outputs)
        

    def run(self):
        """
        Runs the real-time system, managing the initialization, state updates, and input/output handling.
        """
        self.initialize()
        state = self.get_state()
        self.event_manager.set_system_initialized(self)
        while True:
            self.event_manager.wait_system_ready(self)
            self.event_manager.clear_system_ready(self)

            if self.has_feedthrough:
                input_ = self.pull_input_steps()
                self.update_and_push_output(state, input_)
            else:
                self.update_and_push_output(state, None)
                input_ = self.pull_input_steps()
            state = self.debug_wrapper(self.update_state, 'update_state', state, input_)
```

There are a few places to load information into `RealTimeSystem` classes. The `__init__` method is the place to define variable parameters or information that is passed to the `RealTimeSystem`. However, each `RealTimeSystem` runs on a Python multiprocessing process, and not all information can be passed to the process. If you are trying to pass something like a Python `.pickle` file or TensorFlow model, the path can be passed to `__init__` and then the object itself should be loaded in the `initialize` method. Refer to the `mlo.TensorflowModel` class for an example of this.

The `initialize` method can also be used to load the database helper class. `RealTimeSystems` by default are only fed the current information. If you want past information, you can either create a property that tracks the history or you can pull directly from the database. Refer to the `dataio.Historian` class for an example of this.

All `RealTimeSystem` classes run the `run` method. Within this method, the two main functions they can perform are `update_state` and `update_output` (run as part of `update_and_push_output`). The way they perform these two functions depends on a property called `has_feedthrough`, which can be either `True` or `False`. 

As an explanatory example, consider a dynamic system:

$$ x_{k+1}^G = Ax_k^G + Bu_k, \quad y_k = Cx_k^G $$

and a controller:

$$ x_{k+1}^C = Ax_k^C + By_k, \quad u_k = Cx_k^C + Dy_k $$

Considering just inputs and outputs in this system, the dynamic system requires the output of the controller to run its functions, and the controller requires the output of the dynamic system to run its functions. This appears to be an algebraic loop.

What actually should happen is:

1. The dynamic system has `has_feedthrough = False`, so it can run its output equation first to get $y_k$.
1. Then the controller can run both its equations to get the updated controller state and actuator value.
1. Finally, the dynamic system can update its state in preparation for the next time step.

The controller has `has_feedthrough = True` (based on the fact that the D matrix is non-zero), so it must wait for its inputs to be ready. This is what the `has_feedthrough` flag does: it removes seeming algebraic loops that are created when dynamic systems are present.

In the code above, if the system has feedthrough, it must pull inputs before running `update_output`. If it does not have feedthrough, the output equation is not affected by the input and only depends on the state. After either of these are run, the `update_state` function can be run.



