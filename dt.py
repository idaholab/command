# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import copy
import pexpect

import numpy as np

from . import base


class RELAP53D(base.RealtimeSystem):
    """
    A class for interfacing with RELAP5-3D for real-time system simulations.

    The RELAP53D class extends the RealtimeSystem class and provides functionality
    to interface with the RELAP5-3D thermal-hydraulic system code for real-time simulations.

    Attributes:
        initial_tr_file_name (str): The file path for the initial transient input file.
        tr_file_name (str): The file path for the transient input file.
        ss_file_name (str): The file path for the steady-state input file.
        restart_file_name (str): The file path for the restart file.
        output_file_name (str): The file path for the output file.
        plt_file_name (str): The file path for the plot file.
        log_file_name (str): The file path for the log file.
        input_variables (list): A list of dictionaries defining the input variables.
        output_variables (list): A list of dictionaries defining the output variables.
        initial_tr_input_variables (list or None): The initial transient input variables.
        tr_input_variables (list or None): The transient input variables.
        tr_input_variables2 (list or None): The modified transient input variables.
        ss_input_variables2 (list or None): The steady-state input variables.
        tr_time_step (float or None): The time step for the transient simulation.
        initial_tr_lines (list or None): The lines of the initial transient input file.
        tr_lines (list or None): The lines of the transient input file.
        ss_lines (list or None): The lines of the steady-state input file.
        restart_number (str or None): The restart number from the output file.
        child (pexpect.spawn or None): The child process for running RELAP5-3D.
        iteration (int): The current iteration count.
        steady_state_iteration (int): The steady-state iteration count.
        steady_state_frequency (int): The frequency of steady-state updates.
    """

    def __init__(self, name=None, inputs=None, outputs=None,
                 input_variables=None, output_variables=None, relap_file_directory='.', **kwargs):
        """
        Initializes the RELAP53D system with the given parameters.

        Examples of input_variables and output_variables:
            input_variables = [
                # card number, word number, reference value associated with each input variable
                {'card_number': 20299101, 'word_number': 2, 'reference_value': 0.0, 'offset': 0.0},  # reactivity
            ]
            output_variables = [
                # location, variable associated with each output variable
                {'location': 0, 'variable': 'rktpow'},  # reactor power
            ]

        Args:
            name (str, optional): The name of the system. Defaults to None.
            inputs (list or None, optional): The input variables for the system. Defaults to None.
            outputs (list or None, optional): The output variables for the system. Defaults to None.
            input_variables (list, optional): A list of dictionaries defining the input variables. Defaults to None.
            output_variables (list, optional): A list of dictionaries defining the output variables. Defaults to None.
            relap_file_directory (str, optional): The directory containing the RELAP5-3D input files. Defaults to '.'.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=False, **kwargs)
        suffix = '2'
        initial_tr_file_name = relap_file_directory + '/initial_transient_input'
        self.initial_tr_file_name = initial_tr_file_name + suffix
        tr_file_name = relap_file_directory + '/transient_input'
        self.tr_file_name = tr_file_name + suffix
        ss_file_name = relap_file_directory + '/steady_state_input'
        self.ss_file_name = ss_file_name + suffix
        restart_file_name = relap_file_directory + '/restart'
        self.restart_file_name = restart_file_name + suffix
        output_file_name = relap_file_directory + '/output'
        self.output_file_name = output_file_name + suffix
        plt_file_name = restart_file_name
        self.plt_file_name = plt_file_name + suffix
        self.log_file_name = relap_file_directory + 'mylog.txt'

        base.safe_copy(f'{initial_tr_file_name}.i', f'{self.initial_tr_file_name}.i')
        base.safe_copy(f'{tr_file_name}.i', f'{self.tr_file_name}.i')
        base.safe_copy(f'{ss_file_name}.i', f'{self.ss_file_name}.i')
        base.safe_copy(f'{restart_file_name}.r', f'{self.restart_file_name}.r')
        base.safe_copy(f'{plt_file_name}.plt', f'{self.plt_file_name}.plt')
        base.safe_copy(f'{output_file_name}.p', f'{self.output_file_name}.p')

        self.input_variables = base.set_default(input_variables, [])
        self.output_variables = base.set_default(output_variables, [])
        self.initial_tr_input_variables = None
        self.tr_input_variables = None
        self.tr_input_variables2 = None
        self.ss_input_variables2 = None
        self.tr_time_step = None
        self.initial_tr_lines = None
        self.tr_lines = None
        self.ss_lines = None
        self.restart_number = None

        self.child = None
        self.iteration = 0
        self.steady_state_iteration = 0
        self.steady_state_frequency = 10

    def compile_system(self):
        tr_input_variables = self.input_variables
        tr_input_variables.extend([
            {'card_number': 200, 'word_number': 1, 'reference_value': None, 'offset': 0},
            {'card_number': 201, 'word_number': 1, 'reference_value': None, 'offset': self.time_step},
            {'card_number': 103, 'word_number': 1, 'reference_value': None, 'offset': 0},
        ])
        initial_tr_input_variables = copy.deepcopy(tr_input_variables)
        self.tr_lines, self.tr_input_variables = self.parse_input_file(f'{self.tr_file_name}.i', tr_input_variables)
        self.initial_tr_lines, self.initial_tr_input_variables = self.parse_input_file(f'{self.initial_tr_file_name}.i',
                                                                                       initial_tr_input_variables)

        tr_fraction = 0.999
        self.tr_time_step = tr_fraction * self.time_step
        self.tr_input_variables2 = copy.deepcopy(self.tr_input_variables)
        self.tr_input_variables2[-2]['offset'] = self.tr_time_step
        ss_input_variables2 = [
            {'card_number': 200, 'word_number': 1, 'reference_value': None, 'offset': self.tr_time_step},
            {'card_number': 201, 'word_number': 1, 'reference_value': None, 'offset': self.time_step},
            {'card_number': 103, 'word_number': 1, 'reference_value': None, 'offset': 0},
        ]
        self.ss_lines, self.ss_input_variables2 = self.parse_input_file(f'{self.ss_file_name}.i', ss_input_variables2)

        self.inputs.extend([self.names_to_variables[base.time_variable]] * 4)

    @staticmethod
    def parse_input_file(input_file_name, input_variables):
        """
        Parses the input file and associates input variables with line numbers.

        Args:
            input_file_name (str): The file path for the input file.
            input_variables (list): A list of dictionaries defining the input variables.

        Returns:
            tuple: A tuple containing the lines of the file and the updated input variables.
        """
        with open(input_file_name, 'r') as f:
            lines = f.readlines()

        word_length = 10
        line_length = 80
        # remove lines that start with *
        lines = [line for line in lines if line[0] != '*']
        # remove content including and after *
        lines = [line[:line.find('*')] for line in lines]
        # pad lines to have 80 chars
        lines = [line.replace('\n', '')[:line_length].ljust(line_length) + '\n' for line in lines]
        # break lines into cells
        lines = [[line[i:i + word_length] for i in range(0, len(line), word_length)] for line in lines]

        card_numbers = [variable['card_number'] for variable in input_variables]
        for line_number, line in enumerate(lines):
            try:
                card_number = float(line[0])
            except ValueError:
                continue
            if card_number in card_numbers:
                # check that value in line matches value in reference_values
                var_idx = card_numbers.index(card_number)
                word_number = input_variables[var_idx]['word_number']
                line_value = float(line[word_number])
                reference_value = input_variables[var_idx]['reference_value']
                if line_value == reference_value or reference_value is None:
                    input_variables[var_idx]['line_number'] = line_number
                else:
                    raise ValueError(f"Value {reference_value} in reference_values "
                                     f"does not match {line_value} in input file")
        return lines, input_variables

    def update_input_file(self, input_file_name, input_variables, lines, input_values):
        """
        Updates the input file with new input values.

        Args:
            input_file_name (str): The file path for the input file.
            input_variables (list): A list of dictionaries defining the input variables.
            lines (list): The lines of the input file.
            input_values (list): The new input values.
        """
        word_length = 10
        lines2 = copy.deepcopy(lines)
        self.restart_number = self.get_restart_number(f'{self.output_file_name}.p')

        for i, input_variable in enumerate(input_variables):
            if input_variable['card_number'] == 103:
                special_variable = input_variables[i]  # input_variables.pop(i)
        lines2[special_variable['line_number']][special_variable['word_number']] = \
            str(self.restart_number).ljust(word_length)

        regular_input_variables = input_variables[:-1]
        for regular_input_variable, input_value in zip(regular_input_variables, input_values):
            # str_val = f"{input_value + regular_input_variable['offset']:.3E}".ljust(word_length)
            str_val = str(float(input_value + regular_input_variable['offset']))
            if 'e' in str_val:
                str_val = f"{input_value + regular_input_variable['offset']:.5E}"
                str_val = str_val.replace('e', '')
                str_val = str_val.ljust(word_length)
            else:
                str_val = str_val[:9].ljust(word_length)
            # print(float(input_value + regular_input_variable['offset']), str_val)
            # lines2[regular_input_variable['line_number']][regular_input_variable['word_number']] = \
            #     str(input_value + regular_input_variable['offset']).ljust(word_length)
            lines2[regular_input_variable['line_number']][regular_input_variable['word_number']] = str_val

        # for input_variable, input_value in zip(input_variables, input_values):
        # lines2[input_variable['line_number']][input_variable['word_number']] = \
        # str(input_value + input_variable['offset']).ljust(word_length)

        lines2 = [''.join(line) for line in lines2]
        with open(input_file_name, 'w') as f:
            f.writelines(lines2)

    def update_input_file_and_run(self, input_file_name_no_suffix, input_variables, lines, input_values):
        """
        Updates the input file with new input values and runs the RELAP5-3D simulation.

        Args:
            input_file_name_no_suffix (str): The file path for the input file without the suffix.
            input_variables (list): A list of dictionaries defining the input variables.
            lines (list): The lines of the input file.
            input_values (list): The new input values.
        """
        self.update_input_file(f'{input_file_name_no_suffix}.i', input_variables, lines, input_values)
        # base.safe_copy(f'{self.output_file_name}.p', f'{self.output_file_name+str(time.time())}.p')
        base.safe_remove(f'{self.output_file_name}.p')
        base.safe_remove(f'{self.plt_file_name}.plt')
        self.child.sendline(f'relap53D -i {input_file_name_no_suffix}.i '
                            f'-o {self.output_file_name}.p '
                            f'-r {self.restart_file_name}.r '
                            f'-tpfdir /apps/herd/relap5_fluids/')
        try:
            self.child.expect('Transient terminated by end of time step cards.', timeout=60)
        except pexpect.TIMEOUT:
            print("Warning: The RELAP process is taking longer than expected. "
                  "Please check the 'mylog.txt' file for potential issues.")
            self.child.expect('Transient terminated by end of time step cards.', timeout=None)

    @staticmethod
    def get_restart_number(output_file_name):
        """
        Retrieves the last restart number from the output file.

        Args:
            output_file_name (str): The file path for the output file.

        Returns:
            str: The last restart number.
        """
        with open(output_file_name, 'r') as f:
            # Transform all text lines into a single array
            lines_array = np.array(f.read().split())

        [i] = np.where(lines_array == '0---Restart')  # Index of restart numbers
        i_rn = i[-1] + 2  # Index of LAST restart number
        i_t = i[-1] + 2  # Index of LAST restart time
        last_restart_number = lines_array[i_rn]
        last_restart_time = lines_array[i_t]

        return last_restart_number

    @staticmethod
    def parse_output_file(output_file_name, output_variables):
        """
        Parses the output file and retrieves the values for the specified output variables.

        Args:
            output_file_name (str): The file path for the output file.
            output_variables (list): A list of dictionaries defining the output variables.

        Returns:
            np.ndarray: The values of the specified output variables.
        """
        with open(output_file_name, 'r') as f:
            # Transform all text lines into a single array
            lines_array = np.array(f.read().split())

        i1 = np.argmax(lines_array == 'plotalf')
        i2 = np.argmax(lines_array == 'plotnum')
        i3 = np.argmax(lines_array == 'plotrec')
        variables = lines_array[i1:i2]
        locations = lines_array[i2:i3]
        data = lines_array[i3:].reshape((-1, variables.size))[-1, :]
        output_values = [
            float(data[np.logical_and(locations == str(variable['location']), variables == variable['variable'])])
            for variable in output_variables
        ]
        return base.make_1d(output_values)

    def initialize(self):
        child = pexpect.spawn('/bin/bash')
        base.safe_remove(self.log_file_name)
        child.logfile = open(self.log_file_name, 'wb')
        child.sendline('module load relap53D/relap53D-execute')
        child.waitnoecho()
        child.sendline('export SINGULARITY_BIND=/apps/herd/,/projects/')
        child.waitnoecho()
        self.child = child

    def update_state(self, state, input_):

        # t = timeit.default_timer()
        tr_input = input_[:len(self.tr_input_variables) - 1]
        ss_input = input_[len(self.tr_input_variables) - 1:]

        if self.iteration == 0:
            self.update_input_file_and_run(self.initial_tr_file_name, self.initial_tr_input_variables,
                                           self.initial_tr_lines, tr_input)
        elif self.steady_state_iteration != 0 and self.iteration != 0:
            self.update_input_file_and_run(self.tr_file_name, self.tr_input_variables, self.tr_lines, tr_input)
        elif self.steady_state_iteration == 0 and self.iteration != 0:
            self.update_input_file_and_run(self.tr_file_name, self.tr_input_variables2, self.tr_lines, tr_input)
            self.update_input_file_and_run(self.ss_file_name, self.ss_input_variables2, self.ss_lines, ss_input)

        # print(timeit.default_timer() - t, ',', os.stat(f'{self.restart_file_name}.r').st_size /  (1024 * 1024))
        # print(f'update_state: {timeit.default_timer() - t:.3f}\n')

        self.steady_state_iteration = (self.steady_state_iteration + 1) % self.steady_state_frequency
        self.iteration = (self.iteration + 1)

        return np.array([])

    def update_output(self, state, input_):
        return self.parse_output_file(f'{self.plt_file_name}.plt', self.output_variables)
