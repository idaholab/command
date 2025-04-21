# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import os
import shutil
import logging
import time
import threading
import multiprocessing
import timeit

import pexpect
import dill
from decimal import Decimal
import operator
import numpy as np
import pandas as pd
import scipy.signal
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from dotenv import dotenv_values
env = dotenv_values(".env")


# Environment variable settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Logger configuration
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variables
time_variable = 'sim_time'
debug_flag = False


class Simulator:
    """
    Main class for setting up and running a simulation.

    A simulation is made from variables and systems. Variables represent data that can be passed back and forth
    between systems. Systems represent functions that can apply transformations to the variable data. The basic steps
    to starting a simulation are:

        from command import base
        if __name__ == '__main__':  # this is required because the simulation uses parallel processing
            s = base.Simulator(time_step=1)  # initialize the Simulator class with a time step of 1 second
            s.add([])  # add variables and systems
            s.compile()  # compile systems
            s.start()  # start the simulation

    In setting up the simulation, variables are represented by their names, which are strings. Systems also have
    names, but are not used beyond bookkeeping. For example, if you want to create two variables, 'variable1' and
    'variable2' and add them together (forming 'variable3'), you would include the following:

        s.add([
           Variables('variable1', 'variable2', 'variable3'),
           Addition(inputs='variable1', inputs2='variable2', outputs='variable3')
       ])

    All Simulator objects contain a 'sim_time' variable and a Scheduler system. 'sim_time' can be used by systems
    that are a function of simulation time. The Scheduler system synchronizes the time and tells systems when to
    start running calculations.

    In addition, the simulator uses a time-series database to store and retrieve data. More information about the
    database can be found in the DatabaseManager class docstring.

    Attributes:
        time_step (float): Default simulation update time step. This can be overwritten for individual systems.
        data_manager (DataManager): Manager that is shared across all workers that acts as in-memory data.
        event_manager (EventManager): Manager that is shared across all workers that synchronizes workers.
        workers (list): List of Worker class objects that enable multithreading and multiprocessing.
        names_to_variables (dict): A dictionary mapping names to variables.
        names_to_systems (dict): A dictionary mapping names to systems.
    """

    def __init__(self, time_step, simulation_speed=1., end_time=np.inf, auto_start_influx=True):
        """
        Initializes the class.

        Args:
            time_step (float): Default simulation update time step. This can be overwritten for individual systems.
            simulation_speed (float): Multiplier for the maximum rate to run the simulation. This is the maximum rate
                because the true rate will depend on the systems used and the time step. For numerically expensive
                systems and a small time-step, the simulator may not be able to keep up with the simulation_rate.
        """
        self.time_step = float(time_step)
        self.end_time = end_time
        self.auto_start_influx = auto_start_influx
        self.data_manager = None
        self.event_manager = None
        self.workers = []
        self.names_to_variables = {}
        self.names_to_systems = {}
        print('Adding systems...')
        self.add([
            Variable(time_variable),
            Scheduler(name='scheduler', outputs=time_variable, time_step=self.time_step,
                      simulation_speed=simulation_speed, end_time=end_time),
        ])

    def add(self, objects):
        """
        Adds objects (both variables and systems) to the simulation.

        Args:
            objects (list): List of Variable and System objects.
        """
        for object_ in flatten(objects):
            if object_.type == 'variable':
                object_bin = self.names_to_variables
            elif object_.type == 'system':
                object_bin = self.names_to_systems
            else:
                raise ValueError("All object types must be either 'system' or 'variable'.")

            if object_.name in object_bin:
                raise ValueError(f"Names for each type (variable and system) must be unique. "
                                 f"'{object_.name}' of type '{object_.type}' already exists.")
            else:
                if object_.name is None:
                    name_prefix = object_.system_type
                    counter = 0
                    new_name = f'{name_prefix}{counter}'
                    while new_name in object_bin:
                        counter += 1
                        new_name = f'{name_prefix}{counter}'
                    object_.name = new_name
                object_bin[object_.name] = object_

    def compile(self):
        """
        Compiles the systems in the simulation. For each system, this includes setting the data_manager and
        event_manager, running the System.compile() method, and assigning a Worker. For real time systems,
        this also includes assigning time_step. If real time systems are not given specific time_steps when
        initialized, the default value provided in __init__ is used.
        """
        print('Compiling simulation...')
        self.create_managers()
        self.compile_systems()
        self.check_for_errors()

    def create_managers(self):
        """
        Creates the data_manager and event_manager objects for synchronization.
        """
        names_to_rt_systems = {name: system for name, system in self.names_to_systems.items() if system.is_real_time}
        self.data_manager = DataManager(names_to_variables=self.names_to_variables)
        self.event_manager = EventManager(names_to_variables=self.names_to_variables,
                                          names_to_systems=names_to_rt_systems)

    def compile_systems(self):
        """
        Adds default time step and managers to each system, and converts variable strings to variable objects.
        """
        for system in self.names_to_systems.values():
            if system.time_step is None:
                system.set_time_step(self.time_step)
            system.set_managers(data_manager=self.data_manager, event_manager=self.event_manager)
            system.compile(names_to_variables=self.names_to_variables)
            self.workers.append(Worker(system, worker_type='process'))
        self.names_to_systems['scheduler'].set_systems(self.names_to_systems)

    def check_for_errors(self):
        """
        Performs error checking.
        """
        all_outputs = flatten([system.outputs for system in self.names_to_systems.values()
                               if system.outputs is not None])

        # checks to make sure outputs are only used in one system
        overlapping_outputs = self.find_overlapping_outputs()
        if not len(overlapping_outputs):
            for (i, j), common_outputs in overlapping_outputs.items():
                print(f"system {i} and system {j} have overlapping outputs: {common_outputs}")

        all_outputs = set(all_outputs)
        for system in self.names_to_systems.values():
            if system.inputs is None:
                continue

            # checks to make sure constants are not used as system outputs
            constant_outputs = get_constants(system.outputs)
            if len(constant_outputs) > 0:
                constant_names = [output.name for output in constant_outputs]
                raise KeyError(f"constants {constant_names} were used as 'outputs' for system '{system.name}' "
                               f"constants cannot be used as system outputs")

            # checks to make sure all system inputs are generated somewhere
            constant_inputs = get_constants(system.inputs)
            system_inputs = set(make_list(system.inputs)) - set(constant_inputs)
            if system_inputs.intersection(all_outputs) != system_inputs:
                temp_names = tuple([variable.name for variable in list(system_inputs - all_outputs)])
                raise KeyError(f"{temp_names} used as 'inputs' for system '{system.name}' "
                               f"were not generated anywhere (i.e., were not defined as 'outputs' for any system)")

    def find_overlapping_outputs(self):
        """
        Finds any systems with overlapping outputs.

        Returns:
            dict: Dictionary mapping tuple of system names to list of overlapping outputs.
        """
        overlapping_outputs = {}

        for i, system1 in enumerate(self.names_to_systems.values()):
            for j, system2 in enumerate(self.names_to_systems.values()):
                if i < j:  # To avoid duplicate comparisons and self-comparison
                    common_outputs = set(system1.outputs) & set(system2.outputs)
                    if common_outputs:
                        if (system1.name, system2.name) not in overlapping_outputs:
                            overlapping_outputs[(system1.name, system2.name)] = list(common_outputs)

        return overlapping_outputs

    def start(self):
        """
        Starts the simulation. This first starts the Influxdb database and then starts each Worker. For this to work,
        the command 'influxd' entered into the Terminal window must start the database program.
        """
        print('Starting influxdb...')
        influxdb = None
        try:
            if self.auto_start_influx:
                influxdb = pexpect.spawn('influxd')
                influxdb.expect('Listening')
            DatabaseManager().clear_bucket()
            print('Starting workers...')
            for worker in self.workers:
                worker.start()
                if worker.system.name == 'scheduler':
                    scheduler = worker
            scheduler.join()
            for worker in self.workers:
                worker.terminate()
        finally:
            if influxdb is not None:
                influxdb.close()


class VariableArray:
    """
    A variable object represents data that can be passed back and forth between systems. In setting up a simulation,
    variables are called by their names (which are strings) and are used in system inputs and outputs arguments. By
    default, all variables are arrays, and singleton variables are just arrays of size 1. Variables can also be
    called by the Variable and Variables classes.

    Attributes:
        name (str): Variable name.
        size (int): Variable array size.
        scalar_names (list): List of individual names. If size is 1, scalar_names contains the list [name].
            If size is larger than 1, scalar_names contains a list of [name + '0', name + '1', ...].
    """

    def __init__(self, name, size=1):
        """
        Initializes the class.

        Args:
            name (str): Variable name.
            size (int): Variable array size.
        """
        self.name = name
        self.size = size
        if self.size == 1:
            self.scalar_names = [self.name]
        else:
            self.scalar_names = [name + str(i) for i in range(size)]
        self.type = 'variable'

    def __repr__(self):
        return f'(Variable: {self.name})'


class Variable(VariableArray):
    """
    Special instance of VariableArray with size of 1. Refer to VariableArray for documentation.
    """
    def __init__(self, name):
        """
        Initializes the class.

        Args:
            name (str): Variable name.
        """
        super().__init__(name=name, size=1)


class Variables:
    """
    Special instance of Variable for defining multiple Variable objects in one call. Refer to Variable for
    documentation.
    """
    def __new__(cls, *names):
        """
        Initializes a list of multiple Variables.

        Args:
            *names (str): Variable length argument of names. Each name in names creates a variable.
                e.g., Variables('name1', 'name2', 'name3')

        Returns:
            (list): List of Variable objects.
        """
        return [Variable(name) for name in names]


class Constant(VariableArray):
    """
    Special instance of VariableArray with a constant value.
    """
    def __init__(self, name, value):
        """
        Initializes the class.

        Args:
            name (str): Variable name.
            value (float or array-like): Value of the constant.
        """
        self.value = make_1d(value)
        super().__init__(name=name, size=self.value.size)


class DataManager:
    """
    Manager that is shared across all workers and acts as in-memory data. The data is stored in a shared dictionary
    called names_to_data, but is only stored for the current time step.

    Attributes:
        names_to_variables (dict): A dictionary mapping names to variables.
        names_to_data (dict): A dictionary mapping names to data.
    """

    def __init__(self, names_to_variables):
        """
        Initializes the class.

        Args:
        names_to_variables (dict): A dictionary mapping names to variables.
        """
        self.names_to_variables = names_to_variables
        self.names_to_data = multiprocessing.Manager().dict()
        constant_variables = get_constants(self.names_to_variables.values())
        for variable in constant_variables:
            self.names_to_data[variable.name] = variable.value

    def push(self, data, variables):
        """
        Pushes data associated with variables to names_to_data. The size of data must equal the sum of the variable
        sizes and their orders must match.

        Args:
            data (np.ndarray): 1d array of data.
            variables (list): List of variables.
        """
        data = make_1d(data)
        idx = 0
        for variable in variables:
            self.names_to_data[variable.name] = data[idx:idx + variable.size]
            idx += variable.size

    def pull(self, variables):
        """
        Pulls data associated with variables from names_to_data. The size of data will equal the sum of the variable
        sizes and their orders will match.

        Args:
            variables (list): List of variables.

        Returns:
            (np.ndarray): 1d array of data.
        """
        if len(variables) == 0:
            return np.array([])
        else:
            return np.concatenate([self.names_to_data[variable.name] for variable in variables])

    def pull_df(self, clock_time):
        """
        Pulls all data from names_to_data and adds them to a dataframe with index equal to clock_time. To work with a
        dataframe, the data arrays are converted to sequences of scalars, each with a unique name.

        Args:
            clock_time (pd.Timestamp): Clock time associated with the start of the time step.

        Returns:
            (pd.DataFrame): Dataframe.
        """
        df_cols = []
        df_data = []
        for name, data in self.names_to_data.items():
            for scalar_name, datum in zip(self.names_to_variables[name].scalar_names, data):
                if datum is not None:
                    df_cols.append(scalar_name)
                    df_data.append(datum)
        return pd.DataFrame([df_data], index=[clock_time], columns=df_cols)

    def clear(self, variables):
        """
        Clears data associated with variables from names_to_data.

        Args:
            variables (list): List of variables.
        """
        for variable in variables:
            self.names_to_data.pop(variable.name, None)


class DatabaseManager:
    """
    Manager that can be called by any worker to interact with the time-series database software. The DataManager only
    stores data for the current time step. The DatabaseManager stores data across the entire simulation run,
    but is slower than DataManager.

    The database implemented is called Influxdb and is required for the simulation to run. The version used in this
    effort is open-source and can be downloaded at https://portal.influxdata.com/downloads/. The user must download
    and install this and must be able to start it from the Terminal program using the command 'influxd'. This is how
    the simulation code starts Influxdb.

    The parameters for interacting with the database should be stored in a .env file and saved in the same folder
    as the input file. These parameters include a token. To generate a token:

        - Download and install Influxdb.
        - Start it using the 'influxd' command.
        - Open a web browser and navigate to http://localhost:8086 (does not require internet).
        - Create a username and password and follow the prompts.
        - During setup, it will provide a token for HTTP requests. Copy this token and paste it in the .env.

    Attributes:
        client (influxdb_client.InfluxDBClient): Influxdb client for interacting with the database.
    """

    def __init__(self):
        """
        Initializes the class.
        """
        self.client = influxdb_client.InfluxDBClient(url=env['url'], token=env['token'], org=env['org'], debug=False)

    def clear_bucket(self):
        """
        Clears the database bucket of all data.
        """
        bucket_id = self.client.buckets_api().find_bucket_by_name(bucket_name=env['bucket'])
        if bucket_id is not None:
            self.client.buckets_api().delete_bucket(bucket=bucket_id)
        self.client.buckets_api().create_bucket(bucket_name=env['bucket'])
        return self

    def push(self, df):
        """
        Pushes data (stored as a dataframe) to the database.

        Args:
            df (pd.DataFrame): Dataframe of time-series data. Dataframe index must be a time-stamp.
        """
        # the data_frame_measurement_name is basically a type field and is not used beyond setting a value
        with self.client.write_api(write_options=SYNCHRONOUS) as write_client:
            write_client.write(bucket=env['bucket'], org=env['org'], record=df, data_frame_measurement_name='sensor')

    def pull(self, **kwargs):
        """
        Pulls data (stored as a dataframe) from the database.

        Args:
            **kwargs: Keyword arguments. The options for the keywords are defined in the generate_query method.

        Returns:
            (pd.DataFrame): Dataframe of time-series data matching the options in kwargs.
        """
        query = self.generate_query(**kwargs)
        df = self.client.query_api().query_data_frame(query, org=env['org'])

        if not df.empty:
            df = df.set_index('_time').drop(columns=['result', 'table', '_start', '_stop', '_measurement'])
        return df

    def pull_new(self, prior_index=None, **kwargs):
        """
        Pulls new data (stored as a dataframe) from the database since the last index.

        Args:
            prior_index (pd.Index, optional): The previous index to compare against.
            **kwargs: Additional keyword arguments for the query.

        Returns:
            pd.DataFrame or None: Dataframe of new time-series data or None if no new data is found.
        """

        assert 'function' not in kwargs
        assert 'start_time' not in kwargs

        if prior_index is not None:
            kwargs['start_time'] = prior_index.max().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        df = self.pull(**kwargs)
        if prior_index is not None:
            df = df.drop(index=prior_index.max())

        df = df.dropna(axis=0)
        if df.empty:
            return None
        else:
            return df

    @staticmethod
    def generate_query(function=None, sensors=None, start_time='1800-01-01', end_time='2200-01-01'):
        """
        Generates an InfluxDB query string based on the specified parameters.

        Args:
            function (dict, optional): Function to apply to the data (e.g., {'type': 'last'}).
            sensors (list or str, optional): List of sensor names or a single sensor name.
            start_time (str, optional): Start time for the query range.
            end_time (str, optional): End time for the query range.

        Returns:
            str: The generated query string.
        """

        function_filter = ''
        if function is not None:
            if function['type'] == 'first':
                function_filter = '|> first()'
            elif function['type'] == 'last':
                function_filter = '|> last()'
            elif function['type'] == 'tail':
                function_filter = f"|> tail(n:{function['n']}, offset:{function['offset']})"
            else:
                raise ValueError('function must be one of None, first, last, or tail')

        sensor_filter = ''
        if sensors is not None:
            sensors = make_list(sensors)
            sensor_filter += '|> filter(fn: (r) => '
            for i, sensor in enumerate(sensors):
                sensor_filter += f'r["_field"] == "{sensor}"'
                if i < len(sensors) - 1:
                    sensor_filter += ' or '
            sensor_filter += ')'

        query = """
        from(bucket: "{bucket}")
            |> range(start: {start_time}, stop: {end_time})
            {function_filter}
            {sensor_filter}
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        """.format(
            bucket=env['bucket'],
            start_time=start_time,
            end_time=end_time,
            function_filter=function_filter,
            sensor_filter=sensor_filter
        )
        return query

    @staticmethod
    def check_nan(df):
        """
        Checks if the dataframe contains any NaN values.

        Args:
            df (pd.DataFrame): Dataframe to check for NaN values.

        Returns:
            bool: True if the dataframe contains NaN values, False otherwise.
        """
        return np.isnan(df.to_numpy()).sum() > 0


class EventManagerBase:
    """
    A base class for managing events using multiprocessing.

    Attributes:
        items (dict_values): The items to manage events for.
        events (dict): A dictionary mapping item names to multiprocessing events.
    """

    def __init__(self, items):
        """
        Initializes the EventManagerBase with a set of items.

        Args:
            items (dict): A dictionary of items to manage events for.
        """
        self.items = items.values()
        self.events = {name: multiprocessing.Event() for name in items.keys()}

    def __repr__(self):
        """
        Returns a string representation of the event statuses.

        Returns:
            str: A string representation of the event statuses.
        """
        return f'{[(name, event.is_set()) for name, event in self.events.items()]}'

    def check_items(self, items):
        """
        Checks and returns the list of items to manage.

        Args:
            items (str or list): The items to check.

        Returns:
            list: The list of items to manage.
        """
        if items == 'all':
            return self.items
        else:
            return make_list(items)

    def set(self, items):
        """
        Sets the events for the specified items.

        Args:
            items (str or list): The items to set events for.
        """
        items = self.check_items(items)
        for item in items:
            self.events[item.name].set()

    def wait(self, items):
        """
        Waits for the events for the specified items to be set.

        Args:
            items (str or list): The items to wait for.
        """
        items = self.check_items(items)
        for item in items:
            self.events[item.name].wait()

    def clear(self, items):
        """
        Clears the events for the specified items.

        Args:
            items (str or list): The items to clear events for.
        """
        items = self.check_items(items)
        for item in items:
            self.events[item.name].clear()


class EventManager:
    """
    A class for managing events for systems and variables during a simulation.

    Attributes:
        systems_initialized (EventManagerBase): Event manager for system initialization events.
        systems_ready (EventManagerBase): Event manager for system readiness events.
        pulls_done (EventManagerBase): Event manager for pull completion events.
        variables_done (EventManagerBase): Event manager for variable completion events.
    """

    def __init__(self, names_to_variables, names_to_systems):
        """
        Initializes the EventManager with mappings of names to variables and systems.

        Args:
            names_to_variables (dict): A dictionary mapping names to variables.
            names_to_systems (dict): A dictionary mapping names to systems.
        """
        self.systems_initialized = EventManagerBase(names_to_systems)
        self.systems_ready = EventManagerBase(names_to_systems)
        self.pulls_done = EventManagerBase(names_to_systems)
        self.variables_done = EventManagerBase(names_to_variables)

        # sets constant and unused variables to done
        all_outputs = flatten([system.outputs for system in names_to_systems.values() if system.outputs is not None])
        constant_variables = get_constants(names_to_variables.values())
        unused_variables = set(names_to_variables.values()) - set(all_outputs) - set(constant_variables)
        self.set_variables_done(constant_variables + list(unused_variables))

    def set_system_initialized(self, system):
        """
        Sets the system as initialized.

        Args:
            system (str): The name of the system to set as initialized.
        """
        self.systems_initialized.set(system)

    def wait_systems_initialized(self, systems):
        """
        Waits for the systems to be initialized.

        Args:
            systems (str or list): The systems to wait for initialization.
        """
        self.systems_initialized.wait(systems)

    def set_systems_ready(self, systems):
        """
        Sets the status for systems as ready.

        Args:
            systems (str or list): The systems to set as ready.
        """
        self.systems_ready.set(systems)

    def wait_system_ready(self, system):
        """
        Waits for a system to be ready.

        Args:
            system (str): The system to wait for readiness.
        """
        self.systems_ready.wait(system)

    def clear_system_ready(self, system):
        """
        Clears the event for a system being ready.

        Args:
            system (str): The system to clear the readiness event for.
        """
        self.systems_ready.clear(system)

    def set_pull_done(self, system):
        """
        Sets the event for a pull being done.

        Args:
            system (str): The system to set the pull done event for.
        """
        self.pulls_done.set(system)

    def wait_pulls_done(self, systems):
        """
        Waits for pulls to be done for the specified systems.

        Args:
            systems (str or list): The systems to wait for pull completion.
        """
        self.pulls_done.wait(systems)

    def clear_pulls_done(self, systems):
        """
        Clears the events for pulls being done for the specified systems.

        Args:
            systems (str or list): The systems to clear the pull done events for.
        """
        self.pulls_done.clear(systems)

    def set_variables_done(self, variables):
        """
        Sets the events for variables being done.

        Args:
            variables (str or list): The variables to set as done.
        """
        self.variables_done.set(variables)

    def wait_variables_done(self, variables):
        """
        Waits for variables to be done.

        Args:
            variables (str or list): The variables to wait for completion.
        """
        self.variables_done.wait(variables)

    def clear_variables_done(self, variables):
        """
        Clears the events for variables being done.

        Args:
            variables (str or list): The variables to clear the done events for.
        """
        self.variables_done.clear(variables)


class Worker:
    """
    A class to manage the execution of a system as a separate thread or process.

    Attributes:
        system: The system to be managed and executed by the worker.
        worker_type (str): The type of worker, either 'thread' or 'process'.
        worker: The threading.Thread or multiprocessing.Process instance.
    """

    def __init__(self, system, worker_type='process'):
        """
        Initializes the Worker with a system and a worker type.

        Args:
            system: The system to be managed and executed by the worker.
            worker_type (str, optional): The type of worker to create, either 'thread' or 'process'. Defaults to 'process'.
        """
        self.system = system
        self.worker_type = worker_type
        self.worker = None

    def __repr__(self):
        """
        Returns a string representation of the worker.

        Returns:
            str: A string representation of the worker's system.
        """
        return f'{self.system}'

    def start(self):
        """
        Starts the worker as a thread or process based on the worker type.
        """
        if self.worker_type == 'thread':
            self.worker = threading.Thread(target=self.run)
        elif self.worker_type == 'process':
            self.worker = multiprocessing.Process(target=self.run)
        else:
            raise ValueError('worker_type must be either thread or process')
        self.worker.start()

    def join(self):
        """
        Waits for the worker to terminate.
        """
        self.worker.join()

    def run(self):
        """
        Executes the system's run method.
        """
        self.system.run()
        
    def terminate(self):
        """
        Signals the worker to terminate.
        """
        self.worker.terminate()


class System:
    """
    A base class for systems that can be used in simulations. Other classes should inherit from this base class.

    Attributes:
        name (str): The name of the system.
        inputs (list or None): The input variables for the system.
        outputs (list or None): The output variables for the system.
        time_step (float or None): The time step for the system.
        type (str): The type of the system, default is 'system'.
        is_real_time (bool): Indicates if the system runs in real-time, default is False.
        system_type (str): The class name of the system.
        names_to_variables (dict or None): A dictionary mapping names to variables.
        input_size (int or None): The total size of the input variables.
        output_size (int or None): The total size of the output variables.
        parameters (list or None): The parameters for the system.
        data_manager (object or None): The data manager for the system.
        event_manager (object or None): The event manager for the system.
        database_manager (object or None): The database manager for the system.
    """

    def __init__(self, name=None, inputs=None, outputs=None, time_step=None):
        """
        Initializes the System with optional name, inputs, outputs, and time step.

        Args:
            name (str, optional): The name of the system.
            inputs (list or str or None, optional): The input variables for the system.
            outputs (list or str or None, optional): The output variables for the system.
            time_step (float or None, optional): The time step for the system.
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.time_step = time_step
        self.type = 'system'
        self.is_real_time = False
        self.system_type = self.__class__.__name__
        self.names_to_variables = None
        self.input_size = None
        self.output_size = None
        self.parameters = None
        self.data_manager = None
        self.event_manager = None
        self.database_manager = None

    def __repr__(self):
        """
        Returns a string representation of the system.

        Returns:
            str: A string representation of the system.
        """
        return f'(System: {self.name})'
        # return f'{self.system_type}\n\tName: {self.name}\n\tInputs: {self.inputs}\n\tOutputs: {self.outputs}'

    def set_time_step(self, time_step):
        """
        Sets the time step for the system.

        Args:
            time_step (float): The time step for the simulation.
        """
        self.time_step = time_step

    def set_managers(self, data_manager=None, event_manager=None):
        """
        Sets the data manager and event manager for the system.

        Args:
            data_manager (object, optional): The data manager for the system.
            event_manager (object, optional): The event manager for the system.
        """
        self.data_manager = data_manager
        self.event_manager = event_manager

    def compile_variables(self):
        """
        Compiles the input, output, and parameter variables for the system.
        """
        self.inputs = map_names_to_variables(self.inputs, self.names_to_variables, self.name, 'inputs')
        self.outputs = map_names_to_variables(self.outputs, self.names_to_variables, self.name, 'outputs')
        self.parameters = map_names_to_variables(self.parameters, self.names_to_variables, self.name, 'parameters')
        self.input_size = variables_size(self.inputs)
        self.output_size = variables_size(self.outputs)

    def compile_system(self):
        """
        Placeholder method for compiling the system. Should be overridden by subclasses.
        """
        pass

    def compile(self, names_to_variables):
        """
        Compiles the system with the given names to variables mapping.

        Args:
            names_to_variables (dict): A dictionary mapping names to variables.
        """
        self.names_to_variables = names_to_variables
        self.compile_variables()
        self.compile_system()

    def run(self):
        """
        Placeholder method for running the system. Should be overridden by subclasses.
        """
        pass


class Scheduler(System):
    """
    A class for managing the scheduling of real-time systems within a simulation.

    The Scheduler class is responsible for synchronizing the execution of real-time systems, ensuring that they run at the correct intervals, and coordinating the communication between systems and variables.

    Attributes:
        time_step (float): The time step for the scheduler.
        simulation_speed (float): The speed multiplier for the simulation.
        end_time (float): The end time for the simulation.
        min_sleep_time (float): The minimum sleep time for the scheduler loop.
        start_time (pd.Timestamp or None): The start time of the simulation.
        clock_time (pd.Timestamp or None): The current clock time of the simulation.
        run_time (float): The current run time of the simulation.
        names_to_systems (dict or None): A dictionary mapping system names to System objects.
        names_to_rt_systems (dict or None): A dictionary mapping real-time system names to RealtimeSystem objects.
        system_index (dict or None): A dictionary mapping system names to their current index.
    """

    def __init__(self, outputs, time_step, name=None, simulation_speed=1., end_time=np.inf):
        """
        Initializes the Scheduler with the given outputs, time step, and optional name, simulation speed, and end time.

        Args:
            outputs (list or str): The output variables for the scheduler.
            time_step (float): The time step for the scheduler.
            name (str, optional): The name of the scheduler. Defaults to None.
            simulation_speed (float, optional): The speed multiplier for the simulation. Defaults to 1.
            end_time (float, optional): The end time for the simulation. Defaults to np.inf.
        """
        super().__init__(name=name, outputs=outputs)
        self.time_step = time_step
        self.simulation_speed = simulation_speed
        self.end_time = end_time
        self.min_sleep_time = 1e-3
        self.start_time = None
        self.clock_time = None
        self.run_time = 0.
        self.names_to_systems = None
        self.names_to_rt_systems = None
        self.system_index = None

    def set_systems(self, names_to_systems):
        """
        Sets the systems for the scheduler.

        Args:
            names_to_systems (dict): A dictionary mapping system names to System objects.
        """
        self.names_to_systems = names_to_systems
        self.names_to_rt_systems = {name: system for name, system in names_to_systems.items() if system.is_real_time}
        self.system_index = {system.name: -1 for system in self.names_to_rt_systems.values()}

    def calculate_what_to_update(self):
        """
        Calculates which systems and variables need to be updated based on the current run time.

        Returns:
            tuple: A tuple containing a list of systems to update and a list of variables to update.
        """
        systems_to_update = []
        variables_to_update = []
        for system in self.names_to_rt_systems.values():
            temp_index = int(self.make_decimal_class(self.run_time) // self.make_decimal_class(system.time_step))
            if temp_index > self.system_index[system.name]:
                self.system_index[system.name] = temp_index
                systems_to_update.append(system)
                variables_to_update.extend(system.outputs)
        variables_to_update.append(self.names_to_variables[time_variable])
        return systems_to_update, variables_to_update

    def start_simulation_time(self):
        """
        Starts the simulation time by setting the start time and clock time to the current timestamp.
        """
        self.start_time = pd.Timestamp.now()
        self.clock_time = self.start_time

    def update_time(self):
        """
        Updates the run time by adding the time step to the current run time.
        """
        self.run_time = float(self.make_decimal_class(self.run_time) + self.make_decimal_class(self.time_step))

    def wait_time(self):
        """
        Waits until the correct simulation time has elapsed based on the simulation speed.
        """
        while (pd.Timestamp.now() - self.start_time).total_seconds() < (self.run_time / self.simulation_speed):
            remaining_time = (self.run_time / self.simulation_speed - (pd.Timestamp.now() - self.start_time).total_seconds())
            if remaining_time >= self.min_sleep_time:
                time.sleep(0.9 * remaining_time)
            continue
        self.clock_time = pd.Timestamp.now()

    @staticmethod
    def make_decimal_class(value):
        """
        Converts a value to a Decimal class instance.

        Args:
            value: The value to convert. If None, returns None.

        Returns:
            Decimal or None: The value converted to a Decimal instance, or None if the value is None.
        """
        if value is None:
            return value
        else:
            return Decimal(str(value))

    def run(self):
        """
        Runs the scheduler, coordinating the execution of real-time systems and managing data updates.
        """
        self.database_manager = DatabaseManager()
        self.event_manager.wait_systems_initialized('all')
        self.start_simulation_time()
        while True:
            if debug_flag:
                print(f"Start of time step {self.run_time} seconds.")

            systems_to_update, variables_to_update = self.calculate_what_to_update()
            self.event_manager.clear_variables_done(variables_to_update)
            self.data_manager.clear(variables_to_update)
            self.event_manager.clear_pulls_done(systems_to_update)
            self.event_manager.set_systems_ready(systems_to_update)
            self.data_manager.push(self.run_time, self.outputs)
            self.event_manager.set_variables_done(self.outputs)
            self.event_manager.wait_pulls_done('all')
            self.event_manager.wait_variables_done('all')

            df = self.data_manager.pull_df(self.clock_time)
            self.database_manager.push(df)

            self.update_time()
            
            if self.run_time > self.end_time + self.time_step:
                print('Simulation end_time has been met. Simulation is ending.')
                return
            
            self.wait_time()


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
            state (np.ndarray or None): The current state of the system.
            input_ (np.ndarray or None): The input to the system.

        Returns:
            np.ndarray: The updated state of the system.
        """
        return np.array([])

    def update_output(self, state, input_):
        """
        Updates the output of the system based on the current state and input.

        Args:
            state (np.ndarray or None): The current state of the system.
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
        output = make_1d(output)
        self.debug_wrapper(self.data_manager.push, 'data_manager.push', output, self.outputs)
        self.event_manager.set_variables_done(self.outputs)
        
    def debug_wrapper(self, func, step_name, *args, **kwargs):
        """
        Wraps a function call with optional debugging features such as timing and logging.

        If debugging is enabled (indicated by the `debug_flag` variable), this method measures the execution time
        of the function, logs the elapsed time along with the system's name and the step name, and then returns
        the result of the function call. If debugging is disabled, it simply calls the function and returns its result.

        Args:
            func (callable): The function to be wrapped and called.
            step_name (str): The name of the step or function being executed, used for logging purposes.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            The result of the function call.
        """
        if debug_flag:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            print(f"  System: {self.name}, Step: {step_name}, Elapsed time: {round(elapsed_time, 3)}")
            return result
        else:
            return func(*args, **kwargs)

    def run(self):
        """
        Runs the real-time system, managing the initialization, state updates, and input/output handling.
        """
        self.initialize()
        state = self.get_state()
        state = make_1d(state)
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
            state = make_1d(state)


class BasicMath(RealtimeSystem):
    """
    A base class for basic mathematical operations in real-time systems.

    The BasicMath class extends the RealtimeSystem class by adding functionality for basic mathematical operations that involve two sets of inputs. It provides methods to compile variables and systems and to update outputs based on a given mathematical function.

    Attributes:
        inputs2 (list or str): The second set of input variables for the system.
        input_size1 (int or None): The size of the first set of input variables.
        input_size2 (int or None): The size of the second set of input variables.
    """

    def __init__(self, inputs, inputs2, outputs, name=None, **kwargs):
        """
        Initializes the BasicMath system with the given inputs, inputs2, outputs, and optional name and additional keyword arguments.

        Args:
            inputs (list or str): The first set of input variables for the system.
            inputs2 (list or str): The second set of input variables for the system.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.inputs2 = inputs2
        self.input_size1 = None
        self.input_size2 = None

    def compile_variables(self):
        super().compile_variables()
        self.inputs2 = map_names_to_variables(self.inputs2, self.names_to_variables, self.name, 'inputs2')

    def compile_system(self):
        self.input_size1 = variables_size(self.inputs)
        self.input_size2 = variables_size(self.inputs2)
        self.inputs = self.inputs + self.inputs2

    def update_output_base(self, func, state, input_):
        return func(input_[:self.input_size1], input_[self.input_size1:])


class Addition(BasicMath):
    """
    A class for performing addition operations in real-time systems.

    The Addition class extends the BasicMath class and implements the update_output method to perform element-wise addition of inputs.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.add, state, input_)


class Subtraction(BasicMath):
    """
    A class for performing subtraction operations in real-time systems.

    The Subtraction class extends the BasicMath class and implements the update_output method to perform element-wise subtraction of inputs.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.subtract, state, input_)


class Multiplication(BasicMath):
    """
    A class for performing multiplication operations in real-time systems.

    The Multiplication class extends the BasicMath class and implements the update_output method to perform element-wise multiplication of inputs.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.multiply, state, input_)


class Division(BasicMath):
    """
    A class for performing division operations in real-time systems.

    The Division class extends the BasicMath class and implements the update_output method to perform element-wise division of inputs.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.divide, state, input_)


class Average(RealtimeSystem):
    """
    A class for averaging numbers in real-time systems.

    The Average class extends the RealtimeSystem class by adding functionality for calculating the average of the input
    values.
    """

    def __init__(self, inputs, outputs, name=None, **kwargs):
        """
        Initializes the Average system with the given inputs, outputs, and optional name and additional keyword arguments.

        Args:
            inputs (list or str): The first set of input variables for the system.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)

    def update_output(self, state, input_):
        return make_1d(np.mean(input_))


class PeriodicWave(RealtimeSystem):
    """
    A base class for generating periodic waveforms in real-time systems.

    The PeriodicWave class extends the RealtimeSystem class by adding functionality for generating periodic waveforms such as sine, cosine, square, sawtooth, and triangle waves. It provides methods to compile the system with default parameters and to update outputs based on a given waveform function.

    Attributes:
        period (np.ndarray or float or None): The period of the waveform.
        phase (np.ndarray or float or None): The phase shift of the waveform.
        amplitude (np.ndarray or float or None): The amplitude of the waveform.
        bias (np.ndarray or float or None): The bias of the waveform.
    """

    def __init__(self, inputs, outputs, name=None, period=None, phase=None,
                 amplitude=None, bias=None, **kwargs):
        """
        Initializes the PeriodicWave system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            period (np.ndarray or float or None, optional): The period of the waveform. Defaults to None.
            phase (np.ndarray or float or None, optional): The phase shift of the waveform. Defaults to None.
            amplitude (np.ndarray or float or None, optional): The amplitude of the waveform. Defaults to None.
            bias (np.ndarray or float or None, optional): The bias of the waveform. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.period = period
        self.phase = phase
        self.amplitude = amplitude
        self.bias = bias

    def compile_system(self):
        self.period = set_default_array(value=self.period, default=2 * np.pi)
        self.phase = set_default_array(value=self.phase, default=0)
        self.amplitude = set_default_array(value=self.amplitude, default=1)
        self.bias = set_default_array(value=self.bias, default=0)

    def update_output_base(self, func, state, input_):
        function_output = func(2 * np.pi / self.period * (input_ + self.phase))
        return self.amplitude * function_output * np.ones(self.output_size) + self.bias


class SineWave(PeriodicWave):
    """
    A class for generating sine waveforms in real-time systems.

    The SineWave class extends the PeriodicWave class and implements the update_output method to generate sine waveforms.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.sin, state, input_)


class CosineWave(PeriodicWave):
    """
    A class for generating cosine waveforms in real-time systems.

    The CosineWave class extends the PeriodicWave class and implements the update_output method to generate cosine waveforms.
    """
    def update_output(self, state, input_):
        return self.update_output_base(np.cos, state, input_)


class SquareWave(PeriodicWave):
    """
    A class for generating square waveforms in real-time systems.

    The SquareWave class extends the PeriodicWave class and implements the update_output method to generate square waveforms using scipy.signal.square.
    """
    def update_output(self, state, input_):
        return self.update_output_base(scipy.signal.square, state, input_)


class SawtoothWave(PeriodicWave):
    """
    A class for generating sawtooth waveforms in real-time systems.

    The SawtoothWave class extends the PeriodicWave class and implements the update_output method to generate sawtooth waveforms using scipy.signal.sawtooth.
    """
    def update_output(self, state, input_):
        return self.update_output_base(scipy.signal.sawtooth, state, input_)


class TriangleWave(PeriodicWave):
    """
    A class for generating triangle waveforms in real-time systems.

    The TriangleWave class extends the PeriodicWave class and implements the update_output method to generate triangle waveforms using a modified scipy.signal.sawtooth function.
    """
    def update_output(self, state, input_):
        return self.update_output_base(lambda x: scipy.signal.sawtooth(x, width=0.5), state, input_)


class LogicalOperator(RealtimeSystem):
    """
    A class for performing logical operations in real-time systems.

    The LogicalOperator class extends the RealtimeSystem class by adding functionality for evaluating logical conditions between two components and returning one of two possible outputs based on the result of the logical operation.

    Attributes:
        operator (callable): The logical operator function (e.g., greater than, less than, equal to).
    """
    def __init__(self, comp1, op, comp2, outputs, name=None, return1=None, return2=None, **kwargs):
        """
        Initializes the LogicalOperator system with the given components, operator, outputs, and optional parameters.

        Args:
            comp1 (list or str): The first component for the logical comparison.
            op (str): The logical operator to use (e.g., '>', '<', '>=', '<=', '==', '!=').
            comp2 (list or str): The second component for the logical comparison.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            return1 (list or str or None, optional): The output to return if the comparison is True. Defaults to comp1.
            return2 (list or str or None, optional): The output to return if the comparison is False. Defaults to comp2.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.

        Raises:
            KeyError: If the provided operator is not one of the allowed operators.
        """
        operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }
        if op in operators:
            self.operator = operators[op]
        else:
            raise KeyError(f"'op' must be one of {list(operators.keys())}")
        return1 = set_default(return1, comp1)
        return2 = set_default(return2, comp2)
        inputs = [comp1, comp2, return1, return2]
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)

    def update_output(self, state, input_):
        comp1, comp2, return1, return2 = input_
        bool_ = self.operator(comp1, comp2)
        if bool_:
            return return1
        else:
            return return2


class Random(RealtimeSystem):
    """
    A class for generating random numbers from a specified distribution in real-time systems.

    The Random class extends the RealtimeSystem class by adding functionality for generating random numbers from a specified distribution using numpy's random number generator.

    Attributes:
        dist_kwargs (dict): The keyword arguments for the specified distribution method.
        method (callable): The method for generating random numbers from the specified distribution.
    """
    def __init__(self, outputs, dist, name=None, dist_kwargs=None, seed=0, **kwargs):
        """
        Initializes the Random system with the given outputs, distribution, and optional parameters.

        Args:
            outputs (list or str): The output variables for the system.
            dist (str): The name of the distribution method to use (e.g., 'normal', 'uniform').
            name (str, optional): The name of the system. Defaults to None.
            dist_kwargs (dict or None, optional): The keyword arguments for the distribution method. Defaults to None.
            seed (int, optional): The seed for the random number generator. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, outputs=outputs, has_feedthrough=True, **kwargs)
        rng = np.random.default_rng(seed=seed)
        self.dist_kwargs = set_default(dist_kwargs, {})
        self.method = getattr(rng, dist)

    def update_output(self, state, input_):
        return self.method(**self.dist_kwargs)


class AddGaussianNoise(RealtimeSystem):
    """
    A class for adding Gaussian noise to inputs in real-time systems.

    The AddGaussianNoise class extends the RealtimeSystem class by adding functionality for adding Gaussian noise to the input variables.

    Attributes:
        std (np.ndarray or float): The standard deviation of the Gaussian noise.
    """
    def __init__(self, inputs, outputs, std, name=None, **kwargs):
        """
        Initializes the AddGaussianNoise system with the given inputs, outputs, and standard deviation.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            std (np.ndarray or float): The standard deviation of the Gaussian noise.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.std = std

    def compile_system(self):
        self.std = check_size(make_1d(self.std), self.input_size)

    def update_output(self, state, input_):
        return input_ + self.std * np.random.normal(size=(self.input_size,))


class AddRamp(RealtimeSystem):
    """
    A class for adding a ramp function to inputs in real-time systems.

    The AddRamp class extends the RealtimeSystem class by adding functionality for adding a ramp function to the input variables based on the simulation time.

    Attributes:
        slope (float): The slope of the ramp function.
        start_time (float): The start time of the ramp function.
        stop_time (float): The stop time of the ramp function.
    """

    def __init__(self, inputs, outputs, name=None, slope=1, start_time=0, stop_time=np.inf, **kwargs):
        """
        Initializes the AddRamp system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            name (str, optional): The name of the system. Defaults to None.
            slope (float, optional): The slope of the ramp function. Defaults to 1.
            start_time (float, optional): The start time of the ramp function. Defaults to 0.
            stop_time (float, optional): The stop time of the ramp function. Defaults to np.inf.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.slope = slope
        self.start_time = start_time
        self.stop_time = stop_time

    def compile_system(self):
        self.inputs.append(self.names_to_variables[time_variable])

    def update_output(self, state, input_):
        if input_[-1] < self.start_time or input_[-1] >= self.stop_time:
            return input_[:-1]
        else:
            return input_[:-1] + self.slope * (input_[-1] - self.start_time)


class UserDefined(RealtimeSystem):
    """
    A class for applying a user-defined function to inputs in real-time systems.

    The UserDefined class extends the RealtimeSystem class by adding functionality for applying a user-defined function to the input variables.

    Attributes:
        function (callable): The user-defined function to apply to the input variables.
        args (tuple or None): Additional arguments to pass to the user-defined function.
    """

    def __init__(self, name=None, inputs=None, outputs=None, function=None, args=None, **kwargs):
        """
        Initializes the UserDefined system with the given inputs, outputs, function, and optional parameters.

        Args:
            name (str, optional): The name of the system. Defaults to None.
            inputs (list or str or None, optional): The input variables for the system. Defaults to None.
            outputs (list or str or None, optional): The output variables for the system. Defaults to None.
            function (callable or None, optional): The user-defined function to apply to the input variables. Defaults to a lambda function that returns the input.
            args (tuple or None, optional): Additional arguments to pass to the user-defined function. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.function = set_default(function, lambda x: x)
        self.args = args

    def update_output(self, state, input_):
        if self.args is None:
            return make_1d(self.function(input_))
        else:
            return make_1d(self.function(input_, self.args))


class PickleFunction(RealtimeSystem):
    """
    A class for applying a function loaded from a pickle file to inputs in real-time systems.

    The PickleFunction class extends the RealtimeSystem class by adding functionality for applying a function loaded from a pickle file to the input variables.

    Attributes:
        file_name (str): The name of the pickle file containing the function.
        window_size (int): The size of the input window for the function.
        window (np.ndarray or None): The input window for the function.
        function (callable or None): The function loaded from the pickle file.
    """
    def __init__(self, inputs, outputs, file_name, name=None, window_size=1, **kwargs):
        """
        Initializes the PickleFunction system with the given inputs, outputs, file name, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            file_name (str): The name of the pickle file containing the function.
            name (str, optional): The name of the system. Defaults to None.
            window_size (int, optional): The size of the input window for the function. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.file_name = file_name
        self.window_size = window_size
        self.window = None
        self.function = None

    def compile_system(self):
        self.window = np.nan * np.zeros(self.window_size * self.input_size)

    def initialize(self):
        with open(self.file_name, 'rb') as f:
            self.function = dill.load(f)

    def update_output(self, state, input_):
        self.window = slide(window=self.window, new_data=input_)
        if np.any(np.isnan(self.window)):
            return 0.
        return self.function(self.window)


class RateLimiter(RealtimeSystem):
    """
    A class for limiting the rate of change of inputs in real-time systems.

    The RateLimiter class extends the RealtimeSystem class by adding functionality for limiting the rate of change of the input variables, ensuring that the output does not exceed a specified rate of change.

    Attributes:
        rate (float): The maximum rate of change for the input variables.
        minimum (float): The minimum allowable value for the input variables.
        maximum (float): The maximum allowable value for the input variables.
        current_value (float or None): The current value of the output.
    """

    def __init__(self, inputs, outputs, rate, name=None, maximum=np.inf, minimum=-np.inf, **kwargs):
        """
        Initializes the RateLimiter system with the given inputs, outputs, rate, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            rate (float): The maximum rate of change for the input variables.
            name (str, optional): The name of the system. Defaults to None.
            maximum (float, optional): The maximum allowable value for the input variables. Defaults to np.inf.
            minimum (float, optional): The minimum allowable value for the input variables. Defaults to -np.inf.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.rate = rate
        self.minimum = minimum
        self.maximum = maximum
        self.current_value = None

    def update_output(self, state, input_):
        desired_value = input_[0]
        if self.current_value is None:
            self.current_value = desired_value

        if desired_value >= self.maximum:
            desired_value = self.maximum
        elif desired_value <= self.minimum:
            desired_value = self.minimum

        if abs(desired_value - self.current_value) / self.time_step > self.rate:
            self.current_value = self.current_value + np.sign(desired_value - self.current_value) * self.rate * self.time_step
        else:
            self.current_value = desired_value

        return self.current_value


class Sleep(RealtimeSystem):
    """
    A class for introducing a delay in real-time systems.

    The Sleep class extends the RealtimeSystem class by adding functionality for introducing a delay (sleep) in the simulation.

    Attributes:
        seconds (float): The duration of the sleep in seconds.
    """

    def __init__(self, name=None, inputs=None, outputs=None, seconds=0., **kwargs):
        """
        Initializes the Sleep system with the given inputs, outputs, and sleep duration.

        Args:
            name (str, optional): The name of the system. Defaults to None.
            inputs (list or str or None, optional): The input variables for the system. Defaults to None.
            outputs (list or str or None, optional): The output variables for the system. Defaults to None.
            seconds (float, optional): The duration of the sleep in seconds. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.seconds = seconds

    def update_output(self, state, input_):
        time.sleep(self.seconds)
        return [1.0] * self.output_size


class StochasticDelay(RealtimeSystem):
    """
    A class for introducing a stochastic delay in real-time systems.

    The StochasticDelay class extends the RealtimeSystem class by adding functionality for introducing a stochastic delay to the input variables based on a specified distribution.

    Attributes:
        distribution_vals (np.ndarray): The array of possible delay values.
        distribution_probs (np.ndarray): The array of probabilities corresponding to the delay values.
        last_message (np.ndarray or None): The last received message.
        messages (list or None): The list of messages in the delay queue.
        delays (list): The list of delays corresponding to the messages.
    """

    def __init__(self, inputs, outputs, distribution_dict, name=None, **kwargs):
        """
        Initializes the StochasticDelay system with the given inputs, outputs, and distribution dictionary.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            distribution_dict (dict): The dictionary specifying delay values and their probabilities.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.

        Raises:
            ValueError: If any delay values in distribution_dict are negative.
        """
        super().__init__(name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.distribution_vals = np.array(list(distribution_dict.keys())).astype(int)
        if np.any(self.distribution_vals) < 0:
            raise ValueError('The delay values (keys) in distribution_dict must be non-negative.')
        self.distribution_probs = np.array(list(distribution_dict.values())).astype(float)
        self.distribution_probs = self.distribution_probs / np.sum(self.distribution_probs)
        self.last_message = None
        self.messages = None
        self.delays = [0]

    def update_output(self, state, input_):

        if self.messages is None:
            self.last_message = input_[0]
            self.messages = [input_[0]]
        else:
            self.messages.append(input_[0])
            self.delays.append(np.random.choice(self.distribution_vals, p=self.distribution_probs))

        received_flag = True
        received_message = None
        temp_messages = []
        temp_delays = []
        for m, d in zip(self.messages, self.delays):
            if d == 0 and received_flag:
                received_message = m
            else:
                received_flag = False
                temp_messages.append(m)
                temp_delays.append(max(d - 1, 0))

        if received_message is None:
            received_message = self.last_message
        else:
            self.last_message = received_message

        self.messages = temp_messages
        self.delays = temp_delays

        return received_message


def variables_size(variables):
    """
    Calculates the total size of a list of variables.

    Args:
        variables (list): A list of variables, each with a 'size' attribute.

    Returns:
        int: The total size of all variables.
    """
    return int(np.sum([variable.size for variable in variables]))


def get_constants(variables):
    """
    Extracts variables that have a 'value' attribute from a list of variables.

    Args:
        variables (list or list-like): A list of variables to check.

    Returns:
        list: A list of variables that have a 'value' attribute.
    """
    return [variable for variable in variables if hasattr(variable, 'value')]


def make_1d(array):
    """
    Flattens a given array into a 1-dimensional array.

    Args:
        array (array-like): The array to flatten.

    Returns:
        np.ndarray: A 1-dimensional numpy array.
    """
    return np.array(array).flatten()


def make_list(item):
    """
    Converts an item into a list if it is not already a list.

    Args:
        item: The item to convert.

    Returns:
        list: The item converted into a list, or the original item if it is already a list.
    """
    if not isinstance(item, list):
        return [item]
    else:
        return item


def set_default_array(value, default):
    """
    Sets a default array if the given value is None, otherwise flattens the given value to a 1D array.

    Args:
        value (array-like or None): The value to check and possibly flatten.
        default (array-like): The default value to use if the given value is None.

    Returns:
        np.ndarray: The flattened given value, or the flattened default value if the given value is None.
    """
    if value is None:
        return make_1d(default)
    else:
        return make_1d(value)


def set_default(value, default):
    """
    Sets a default value if the given value is None.

    Args:
        value: The value to check.
        default: The default value to use if the given value is None.

    Returns:
        The given value if it is not None, otherwise the default value.
    """
    if value is None:
        return default
    else:
        return value


def check_size(array, size):
    """
    Ensures an array is of the specified size by replicating its elements if necessary.

    Args:
        array (array-like): The array to check.
        size (int): The desired size of the array.

    Returns:
        np.ndarray: An array of the specified size, with replicated elements if necessary.
    """
    return array * np.ones(size)


def map_names_to_variables(names, names_to_variables, system_name, variables_category):
    """
    Maps a list of names to corresponding variables from a dictionary.

    Args:
        names (list or str): The names to map to variables.
        names_to_variables (dict): A dictionary mapping names to variables.
        system_name (str): The name of the system for error messages.
        variables_category (str): The category of variables for error messages.

    Returns:
        list: A list of variables corresponding to the given names.

    Raises:
        KeyError: If a name is not found in the names_to_variables dictionary.
    """
    if names is None:
        return []
    names = make_list(names)

    variables_list = []
    for name in names:
        if name in names_to_variables:
            variables_list.append(names_to_variables[name])
        else:
            raise KeyError(f"'{name}' used as '{variables_category}' for system '{system_name}' "
                           f"was not defined as a variable.")

    return variables_list


def safe_remove(file_name):
    """
    Safely removes a file if it exists.

    Args:
        file_name (str): The name of the file to remove.
    """
    if os.path.exists(file_name):
        os.remove(file_name)


def safe_copy(file_name, new_file_name):
    """
    Safely copies a file to a new location if it exists.

    Args:
        file_name (str): The name of the file to copy.
        new_file_name (str): The name of the new file location.

    Raises:
        FileNotFoundError: If the file to copy does not exist.
    """
    if os.path.exists(file_name):
        shutil.copyfile(file_name, new_file_name)
    else:
        raise FileNotFoundError(f'File {file_name} not found.')


def flatten(items, item_types=(list, tuple)):
    """
    Flattens a nested list or tuple into a single list.

    Args:
        items (list or tuple): The nested list or tuple to flatten.
        item_types (tuple, optional): The types to consider for flattening. Defaults to (list, tuple).

    Returns:
        list: The flattened list or tuple. If the input is None, returns None.
    """
    if items is None:
        return items

    item_type = type(items)
    items = list(items)
    i = 0
    while i < len(items):
        while isinstance(items[i], item_types):
            if not items[i]:
                items.pop(i)
                i -= 1
                break
            else:
                items[i:i + 1] = items[i]
        i += 1
    return item_type(items)


def create_windowed_data(data, window_size, window_spacing=1, swap_axes=False):
    """
    Turns a 2d multivariate time-series array into a 3d windowed multivariate time-series array.

    Args:
        data (np.ndarray): A 2d array of dimension (num_steps, num_vars).
        window_size (int): The num_steps used per window.
        window_spacing (int): The start-to-start spacing between windows. If window_spacing=window_length,
            there is no overlap.
        swap_axes (bool): If True, swaps last two axes.

    Returns:
        (np.ndarray): A 3d windowed array of dimension (num_windows, num_vars, window_length).
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    num_steps = data.shape[0]
    num_vars = data.shape[1]
    window_start = np.arange(0, num_steps, window_spacing)
    window_start = window_start[window_start + window_size <= num_steps]
    num_windows = window_start.size
    windowed_data = np.zeros((num_windows, num_vars, window_size), dtype=data.dtype)
    for i in range(window_size):
        windowed_data[:, :, i] = data[window_start + i, :]
    if swap_axes:
        windowed_data = np.swapaxes(windowed_data, 1, 2)

    return windowed_data


def slide(window, new_data):
    """
    Slides a window of data by appending new data and removing the oldest data.

    Args:
        window (np.ndarray): The current window of data.
        new_data (np.ndarray): The new data to append to the window.

    Returns:
        np.ndarray: The updated window with the new data appended and the oldest data removed.
    """
    return np.concatenate([window[new_data.size:], new_data])
