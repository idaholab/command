# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import pandas as pd
import numpy as np
import dill
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow import keras

from . import base


class AnomalyDetector(base.RealtimeSystem):
    """
    A class for detecting anomalies in real-time systems using historical data and PCA.

    The AnomalyDetector class extends the RealtimeSystem class by adding functionality for detecting anomalies based on historical data and Principal Component Analysis (PCA).

    Attributes:
        historical_data_file (str): The file path to the historical data CSV file.
        pickle_file (str): The file path to the pickle file for the trained PCA model.
        method (str): The anomaly detection method to use (currently only 'pca' is supported).
        pca_var (float): The variance to retain in the PCA model.
        window_size (int): The size of the window for creating windowed data.
        window (np.ndarray or None): The windowed data for the current input.
        function (callable or None): The trained PCA model for scoring anomalies.
        active (bool): Indicates whether the anomaly detector is active.
    """
    def __init__(self, inputs, outputs, historical_data_file, method, name=None, pca_var=0.99, window_size=1,
                 active=True, **kwargs):
        """
        Initializes the AnomalyDetector system with the given inputs, outputs, and optional parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            historical_data_file (str): The file path to the historical data CSV file.
            method (str): The anomaly detection method to use (currently only 'pca' is supported).
            name (str, optional): The name of the system. Defaults to None.
            pca_var (float, optional): The variance to retain in the PCA model. Defaults to 0.99.
            window_size (int, optional): The size of the window for creating windowed data. Defaults to 1.
            active (bool, optional): Indicates whether the anomaly detector is active. Defaults to True.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=False, **kwargs)
        self.historical_data_file = historical_data_file
        self.pickle_file = historical_data_file.replace('.csv', '.pickle')
        self.method = method
        self.pca_var = pca_var
        self.window_size = window_size
        self.window = None
        self.function = None
        self.active = active

    def compile_system(self):
        if not self.active:
            return
        if self.method == 'pca':
            df = pd.read_csv(filepath_or_buffer=self.historical_data_file, header=0, index_col=0,
                             infer_datetime_format=True, parse_dates=True)
            inputs_names = [input_.name for input_ in self.inputs]
            data = df[inputs_names].to_numpy()

            # create windowed data
            data = base.create_windowed_data(data, window_size=self.window_size, window_spacing=1, swap_axes=True)
            data = data.reshape(data.shape[0], -1)

            # scale data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # train model
            pca = PCA(self.pca_var)
            pca.fit(data_scaled)
            # print(pca.explained_variance_ratio_.size)
            data_scaled_hat = pca.inverse_transform(pca.transform(data_scaled))
            error_std = np.std(data_scaled - data_scaled_hat, axis=0)

            def score_function(y):
                if y.ndim == 1:
                    y = y.reshape(1, -1)
                y_scaled = scaler.transform(y)
                y_scaled_hat = pca.inverse_transform(pca.transform(y_scaled))
                scaled_error = (y_scaled - y_scaled_hat) / error_std
                return np.sqrt(np.mean(scaled_error ** 2, axis=1))

            with open(self.pickle_file, 'wb') as f:
                dill.dump(score_function, f, recurse=True)
        else:
            raise ValueError('The only available method is pca')
        self.window = np.nan * np.zeros(self.window_size * self.input_size)

    def initialize(self):
        if not self.active:
            return
        with open(self.pickle_file, 'rb') as f:
            self.function = dill.load(f)

    def get_state(self):
        return 0.

    def update_state(self, state, input_):

        if not self.active:
            return 0.

        self.window = base.slide(window=self.window, new_data=input_)
        if np.any(np.isnan(self.window)):
            # TODO: enable this to return None and just not plot it
            return 0.
        return self.function(self.window)

    def update_output(self, state, input_):
        return state


class TensorflowModel(base.RealtimeSystem):
    """
    A class for applying a TensorFlow model to inputs in real-time systems.

    The TensorflowModel class extends the RealtimeSystem class by adding functionality for applying a pre-trained TensorFlow model to the input variables.

    Attributes:
        file_name (str): The file path to the TensorFlow model.
        model (keras.Model or None): The loaded TensorFlow model.
    """
    def __init__(self, inputs, outputs, file_name, name=None, **kwargs):
        """
        Initializes the TensorflowModel system with the given inputs, outputs, and file name.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            file_name (str): The file path to the TensorFlow model.
            name (str, optional): The name of the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.file_name = file_name
        self.model = None

    def initialize(self):
        self.model = keras.models.load_model(self.file_name)

    def update_output(self, state, input_):
        return base.make_1d(self.model(input_.reshape(1, -1)).numpy())


class TensorflowDynamicModel(base.RealtimeSystem):
    """
    A class for applying a dynamic TensorFlow model to inputs in real-time systems.

    The TensorflowDynamicModel class extends the RealtimeSystem class by adding functionality for applying a pre-trained dynamic TensorFlow model to the input variables, taking into account the system's state and input history.

    Attributes:
        file_name (str): The file path to the TensorFlow model.
        model (keras.Model or None): The loaded TensorFlow model.
        n_z (int): The number of state variables.
        n_u (int): The number of input variables.
        z_window_size (int): The size of the window for the state variables.
        u_window_size (int): The size of the window for the input variables.
        initial_conditions (np.ndarray or None): The initial conditions for the state variables.
    """

    def __init__(self, inputs, outputs, file_name,
                 n_z, n_u, z_window_size, u_window_size,
                 name=None, initial_conditions=None, **kwargs):
        """
        Initializes the TensorflowDynamicModel system with the given inputs, outputs, file name, and other parameters.

        Args:
            inputs (list or str): The input variables for the system.
            outputs (list or str): The output variables for the system.
            file_name (str): The file path to the TensorFlow model.
            n_z (int): The number of state variables.
            n_u (int): The number of input variables.
            z_window_size (int): The size of the window for the state variables.
            u_window_size (int): The size of the window for the input variables.
            name (str, optional): The name of the system. Defaults to None.
            initial_conditions (np.ndarray or None, optional): The initial conditions for the state variables. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        super().__init__(name=name, inputs=inputs, outputs=outputs, **kwargs)
        self.file_name = file_name
        self.model = None
        self.n_z = n_z
        self.n_u = n_u
        self.z_window_size = z_window_size
        self.u_window_size = u_window_size
        self.initial_conditions = initial_conditions

    def initialize(self):
        self.model = keras.models.load_model(self.file_name)

    def get_state(self):
        return self.initial_conditions

    @staticmethod
    def mod_function(window_data, window_size):
        window_data = window_data.reshape((window_size, -1))
        mod_data = [window_data[i, :] if i == 0 else window_data[i - 1, :] - window_data[i, :]
                    for i in range(window_size)]
        return np.array(mod_data).flatten()

    def update_state(self, state, input_):
        # state comes in as [z_k, z_{k-1}, ... , z_{k-pz}, u_{k-1}, u_{k-1}, ... , u_{k-1-pu}]
        # need to remove u_{k-1-pu}, shift right, and insert u_k

        z_window = state[:(self.n_z * self.z_window_size)]
        um1_window = state[(self.n_z * self.z_window_size):]
        u_window = np.concatenate([input_, um1_window[:-self.n_u]])

        z_window_mod = self.mod_function(z_window, self.z_window_size)
        u_window_mod = self.mod_function(u_window, self.u_window_size)

        x = np.concatenate([z_window_mod, u_window_mod])
        z = z_window[:self.n_z]
        zp = z + base.make_1d(self.model(x.reshape(1, -1)).numpy())
        zp_window = np.concatenate([zp, z_window[:-self.n_z]])
        return np.concatenate([zp_window, u_window])

    def update_output(self, state, input_):
        return state[:self.n_z]
