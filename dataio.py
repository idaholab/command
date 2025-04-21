# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import time
import os

import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback
from influxdb_client.rest import ApiException
import plotly.graph_objects as go

from . import base


class Visualization(base.System):
    """
    A class for visualizing real-time data using a web-based dashboard.

    The Visualization class extends the System class and provides functionality
    to display real-time data updates in a web-based dashboard using Plotly Dash.

    Attributes:
        dash_server (str): The URL for the Dash server.
        plot_dicts (list): A list of dictionaries defining the plots to display.
        update_time (int): The interval time (in seconds) for updating the dashboard.
        n_time_steps (int): The number of time steps to include in the plots.
        debug (bool): Indicates whether to run the Dash server in debug mode.
    """

    dash_server = 'http://127.0.0.1:8050'

    def __init__(self, name=None, update_time=1, n_time_steps=500, plot_dicts=None, debug=False):
        """
        Initializes the Visualization system with the given parameters.

        Args:
            name (str, optional): The name of the system. Defaults to None.
            update_time (int, optional): The interval time (in seconds) for updating the dashboard. Defaults to 1.
            n_time_steps (int, optional): The number of time steps to include in the plots. Defaults to 500.
            plot_dicts (list, optional): A list of dictionaries defining the plots to display. Defaults to None.
            debug (bool, optional): Indicates whether to run the Dash server in debug mode. Defaults to False.
        """
        super().__init__(name=name)
        self.plot_dicts = base.set_default(plot_dicts, [None])
        self.update_time = update_time
        self.n_time_steps = n_time_steps
        self.debug = debug

    def get_start_time(self):
        """
        Retrieves the start time of the simulation from the database.

        Returns:
            datetime: The start time of the simulation.
        """
        while True:
            try:
                temp = self.database_manager.pull(function={'type': 'first'}, sensors=base.time_variable)
                start_time = temp.index[0].tz_localize(None)
                return start_time
            except (ApiException, IndexError) as e:
                time.sleep(0.5)

    def prepare_layout(self):
        """
        Prepares the layout for the Dash app.
        """
        layout = [
            dcc.Interval(id='interval-component', interval=1000 * self.update_time, n_intervals=0),
            html.H1('Simulation Dashboard'),
            html.Div(id='live-text'),
        ]

        graph_outputs = []
        variables_lists = []

        for i, plot_dict in enumerate(self.plot_dicts):
            graph_name = f'live-graph{i}'
            graph_outputs.append(Output(graph_name, 'figure'))
            if plot_dict is None:
                variables_lists = None
                description = 'All Variables'
            else:
                variables_lists.append([item for name in plot_dict['variables'] for item in self.names_to_variables[name].scalar_names])
                description = plot_dict.get('description', str(plot_dict['variables']).translate({ord(i): None for i in "'[]"}))
            layout.append(html.H2(description))
            layout.append(dcc.Graph(id=graph_name, animate=True))

        return layout, graph_outputs, variables_lists

    def run(self):
        self.database_manager = base.DatabaseManager()
        start_time = self.get_start_time()

        app = Dash(__name__)
        layout, graph_outputs, variables_lists = self.prepare_layout()
        all_variables = base.flatten(variables_lists)
        if all_variables is not None and base.time_variable not in all_variables:
            all_variables.append(base.time_variable)

        pull_function = {'type': 'tail', 'n': self.n_time_steps, 'offset': 0}
        df = self.database_manager.pull(function=pull_function, sensors=all_variables).dropna()
        df = df.set_index(base.time_variable)
        df.index = pd.to_datetime(df.index, unit='s')

        figs = []
        temp_variables_lists = base.set_default(variables_lists, [df.columns])
        for fig_variables in temp_variables_lists:
            fig = go.Figure()
            for variable in fig_variables:
                fig.add_trace(go.Scatter(x=df.index, y=df[variable], mode='lines', name=variable, showlegend=True))
            fig = self.update_figure_layout(fig, df, fig_variables)
            figs.append(fig)


        @callback([Output('live-text', 'children')] + graph_outputs, Input('interval-component', 'n_intervals'))
        def interval_updates(_):
            pull_function = {'type': 'tail', 'n': self.n_time_steps, 'offset': 0}
            df = self.database_manager.pull(function=pull_function, sensors=all_variables).dropna()

            simulation_time = pd.to_timedelta(df.loc[df.index[-1], base.time_variable].squeeze(), unit='s').round('1s')
            current_time = df.index[-1].tz_localize(None)
            elapsed_time = (current_time - start_time).round('1s')
            html_times = [[html.Div(f'Simulation Time: {simulation_time}'),
                           html.Div(f'Elapsed Time: {elapsed_time}')]]

            df = df.set_index(base.time_variable)
            df.index = pd.to_datetime(df.index, unit='s')

            temp_variables_lists = base.set_default(variables_lists, [df.columns])
            for i, fig_variables in enumerate(temp_variables_lists):
                fig = figs[i]
                for j, variable in enumerate(fig_variables):
                    fig.data[j].x = df.index
                    fig.data[j].y = df[variable]
                fig = self.update_figure_layout(fig, df, fig_variables)
                # figs[i] = fig
            return html_times + figs

        app.layout = html.Div(layout)
        app.run_server(debug=self.debug, use_reloader=False)
        
    def update_figure_layout(self, fig, df, variables):
        """
        Updates the layout of the figure.
        """
        range_multiplier = 0.05
        index_range = pd.Timedelta(f'{self.n_time_steps * self.time_step}s')
        index_min = df.index.min() - range_multiplier * index_range
        index_max = max(df.index.max(), df.index.min() + index_range) + range_multiplier * index_range
        x_range = [index_min, index_max]
        variables_min = df[variables].to_numpy().min()
        variables_max = df[variables].to_numpy().max()
        variables_range = variables_max - variables_min
        y_range = [variables_min - range_multiplier * variables_range, variables_max + range_multiplier * variables_range]
        fig.update_layout(
            xaxis_range=x_range,
            xaxis_tickformat='%H:%M:%S',
            yaxis_range=y_range,
            margin={'t': 10, 'b': 10},
            height=300
        )
        return fig


class Historian(base.RealtimeSystem):
    """
    A class for logging real-time data to a CSV file.

    The Historian class extends the RealtimeSystem class and provides functionality
    to log real-time data to a specified CSV file.

    Attributes:
        path (str): The file path for the CSV file.
        prior_index (pd.Index or None): The index of the last logged data.
    """

    def __init__(self, path, name=None, **kwargs):
        """
        Initializes the Historian system with the given parameters.

        Args:
            path (str): The file path for the CSV file.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, **kwargs)
        self.path = self.check_path(path)
        self.prior_index = None

    @staticmethod
    def check_path(path):
        """
        Checks if the specified file path exists and handles overwriting or renaming.

        Args:
            path (str): The file path for the CSV file.

        Returns:
            str: The final file path.
        """
        if path is not None and os.path.exists(path):
            ans = input('File already exists. Do you want to overwrite this file? (y/n) ')
            if ans.lower() == 'y':
                os.remove(path)
            else:
                name, extension = os.path.splitext(path)
                counter = 1
                while os.path.exists(path):
                    path = f'{name}{counter}{extension}'
                    counter += 1
        return path

    def initialize(self):
        if self.path is None:
            return
        self.database_manager = base.DatabaseManager()

    def update_output(self, state, input_):
        df = self.database_manager.pull_new(self.prior_index, sensors=None)
        if df is not None:
            self.prior_index = df.index
            df = df.set_index(base.time_variable)
            df.to_csv(self.path, mode='a', header=not os.path.exists(self.path))
