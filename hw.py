# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import json
import socket
import time

import grpc
import numpy as np

from . import base, grpc_TB_pb2, grpc_TB_pb2_grpc


class gRPC(base.RealtimeSystem):
    """
    A class for handling gRPC communication in real-time applications.

    The gRPC class extends the RealtimeSystem class by adding functionality for sending and receiving data
    using gRPC protocol.

    This class incorporates triggers, which control the order of operations when using gRPC communication.

    Attributes:
        mode (str): The mode of operation, either 'client' or 'server'.
        ip_address (str): The IP address of the gRPC server.
        port (int): The port number of the gRPC server.
        method_class (str): The gRPC method class to use for communication.
        channels (list or None): The channels for communication.
        proto_metadata (dict): Metadata for the gRPC methods.
        grpc_type (str): The type of gRPC method ('Get' or 'Set').
    """

    def __init__(self, mode, ip_address, port, method_class, name=None, channels=None, inputs=None, outputs=None,
                 triggers=None, **kwargs):
        """
        Initializes the gRPC system with the given parameters.

        Args:
            mode (str): The mode of operation, either 'client' or 'server'.
            ip_address (str): The IP address of the gRPC server.
            port (int): The port number of the gRPC server.
            method_class (str): The gRPC method class to use for communication.
            name (str, optional): The name of the system. Defaults to None.
            channels (list, optional): The channels for communication. Defaults to None.
            inputs (list or None, optional): The input variables for the system. Defaults to None.
            outputs (list or None, optional): The output variables for the system. Defaults to None.
            triggers (list or None, optional): The trigger variables for the system. Defaults to None.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        if mode not in ['client', 'server']:
            raise ValueError("mode must be either 'client' or 'server'")
        if mode == 'server':
            raise ValueError("mode='server' for gRPC system has not been developed yet.")
        if (inputs is None) + (outputs is None) != 1:  # xor
            raise ValueError('Only one of inputs or outputs can be used at a time.')

        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.mode = mode
        self.ip_address = ip_address
        self.port = port
        self.method_class = method_class
        self.channels = channels

        # get method
        if 'Get' in self.method_class and self.outputs is not None:
            self.grpc_type = 'Get'
            self.inputs = triggers
        # set method
        elif 'Set' in self.method_class and self.inputs is not None:
            self.grpc_type = 'Set'
            self.outputs = triggers
        else:
            raise ValueError("Method class, inputs, and outputs are incompatible. If using a 'Get' method class, "
                             "'inputs' must be None and 'outputs' must be non-empty. If using a 'Set' method class, "
                             "'inputs' must be non-empty and 'outputs' must be None.")

        self.define_proto_metadata()
        if self.method_class not in self.proto_metadata:
            raise KeyError(f"Provided method class must be one of '{list(self.proto_metadata.keys())}'")

    def compile_system(self):
        if self.channels is None:
            if self.grpc_type == 'Set':
                self.channels = [input_.name for input_ in self.inputs]
            elif self.grpc_type == 'Get':
                self.channels = [output.name for output in self.outputs]

    def define_proto_metadata(self):
        """
        Defines the metadata for the gRPC methods.
        """
        self.proto_metadata = {
            'GetDouble': {'RequestType': 'DoubleName', 'ReplyType': 'DoubleValue', 'name_field': 'ChannelName',
                          'data_field': 'DoubleChannel'},
            'GetFloat': {'RequestType': 'FloatName', 'ReplyType': 'FloatValue', 'name_field': 'ChannelName',
                         'data_field': 'FloatChannel'},
            'GetInt64': {'RequestType': 'Int64Name', 'ReplyType': 'Int64Value', 'name_field': 'ChannelName',
                         'data_field': 'Int64Channel'},
            'GetInt32': {'RequestType': 'Int32Name', 'ReplyType': 'Int32Value', 'name_field': 'ChannelName',
                         'data_field': 'Int32Channel'},
            'GetUint64': {'RequestType': 'Uint64Name', 'ReplyType': 'Uint64Value', 'name_field': 'ChannelName',
                          'data_field': 'Uint64Channel'},
            'GetUint32': {'RequestType': 'Uint32Name', 'ReplyType': 'Uint32Value', 'name_field': 'ChannelName',
                          'data_field': 'Uint32Channel'},
            'GetBool': {'RequestType': 'BoolName', 'ReplyType': 'BoolValue', 'name_field': 'ChannelName',
                        'data_field': 'BoolChannel'},
            'SetDouble': {'RequestType': 'DoubleValue', 'ReplyType': 'DoubleReply', 'name_field': 'ChannelName',
                          'data_field': 'DoubleChannel'},
            'SetFloat': {'RequestType': 'FloatValue', 'ReplyType': 'FloatReply', 'name_field': 'ChannelName',
                         'data_field': 'FloatChannel'},
            'SetInt64': {'RequestType': 'Int64Value', 'ReplyType': 'Int64Reply', 'name_field': 'ChannelName',
                         'data_field': 'Int64Channel'},
            'SetInt32': {'RequestType': 'Int32Value', 'ReplyType': 'Int32Reply', 'name_field': 'ChannelName',
                         'data_field': 'Int32Channel'},
            'SetUint64': {'RequestType': 'Uint64Value', 'ReplyType': 'Uint64Reply', 'name_field': 'ChannelName',
                          'data_field': 'Uint64Channel'},
            'SetUint32': {'RequestType': 'Uint32Value', 'ReplyType': 'Uint32Reply', 'name_field': 'ChannelName',
                          'data_field': 'Uint32Channel'},
            'SetBool': {'RequestType': 'BoolValue', 'ReplyType': 'BoolReply', 'name_field': 'ChannelName',
                        'data_field': 'BoolChannel'},
            'SetAxisPosition': {'RequestType': 'AxisValue', 'ReplyType': 'AxisReply', 'name_field': 'AxisName',
                                'data_field': 'DoubleValue'}
        }

    def generate_request_payload(self, input_):
        """
        Generates the request payload for the gRPC method.

        Args:
            input_ (np.ndarray): The input data.

        Returns:
            dict: The request payload.
        """
        if self.grpc_type == 'Get':
            return {
                self.proto_metadata[self.method_class]['name_field']: self.channels
            }
        elif self.grpc_type == 'Set':
            return {
                self.proto_metadata[self.method_class]['name_field']: self.channels,
                self.proto_metadata[self.method_class]['data_field']: list(input_)
            }
        else:
            raise ValueError("There is a bug in the code. This should be unreachable.")

    def parse_reply_payload(self, reply):
        """
        Parses the reply payload from the gRPC method.

        Args:
            reply (grpc.Response): The gRPC response.

        Returns:
            np.ndarray: The parsed output data.
        """
        if self.grpc_type == 'Get':
            return base.make_1d(getattr(reply, self.proto_metadata[self.method_class]['data_field']))
        elif self.grpc_type == 'Set':
            return [1.0] * self.input_size  # triggers
        else:
            raise ValueError("There is a bug in the code. This should be unreachable.")

    def update_output(self, state, input_):

        with grpc.insecure_channel(f'{self.ip_address}:{self.port}') as channel:
            stub = grpc_TB_pb2_grpc.gRPCTBStub(channel)
            fields_to_values = self.generate_request_payload(input_)
            request = getattr(grpc_TB_pb2, self.proto_metadata[self.method_class]['RequestType'])(**fields_to_values)
            reply = getattr(stub, self.method_class)(request)
            return self.parse_reply_payload(reply)


class SocketCommunication(base.RealtimeSystem):
    """
    A class for handling socket communication in real-time applications.

    The SocketCommunication class extends the RealtimeSystem class by adding functionality for sending and receiving data
    using socket communication.

    Attributes:
        mode (str): The mode of operation, either 'client' or 'server'.
        protocol (str): The protocol for communication, either 'array', 'scalar', or 'json'.
        ip_address (str): The IP address for the socket communication. Defaults to 'localhost'.
        port (int): The port number for the socket communication. Defaults to 8090.
        server (socket.socket or None): The server socket object.
    """

    def __init__(self, mode, protocol, name=None, inputs=None, outputs=None, ip_address='localhost', port=8090, **kwargs):
        """
        Initializes the SocketCommunication system with the given parameters.

        Args:
            mode (str): The mode of operation, either 'client' or 'server'.
            protocol (str): The protocol for communication, either 'array', 'scalar', or 'json'.
            name (str, optional): The name of the system. Defaults to None.
            inputs (list or None, optional): The input variables for the system. Defaults to None.
            outputs (list or None, optional): The output variables for the system. Defaults to None.
            ip_address (str, optional): The IP address for the socket communication. Defaults to 'localhost'.
            port (int, optional): The port number for the socket communication. Defaults to 8090.
            **kwargs: Additional keyword arguments passed to the RealtimeSystem initializer.
        """
        if mode not in ['client', 'server']:
            raise ValueError("mode must be either 'client' or 'server'")
        if (inputs is None) + (outputs is None) != 1:  # xor
            raise ValueError('Only one of inputs or outputs can be used at a time.')
        if protocol not in ['array', 'scalar', 'json']:
            raise ValueError("protocol must be either 'array', 'scalar', or 'json'")

        super().__init__(name=name, inputs=inputs, outputs=outputs, has_feedthrough=True, **kwargs)
        self.mode = mode
        self.protocol = protocol
        self.ip_address = ip_address
        self.port = port
        self.server = None

    def initialize(self):
        if self.mode == 'server':
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_address = (self.ip_address, self.port)
            self.server.bind(server_address)
            self.server.listen(1)

    def get_connection(self):
        """
        Establishes a connection based on the mode of operation.

        Returns:
            socket.socket: The socket connection object.

        Raises:
            RuntimeError: If the client fails to connect after multiple attempts.
        """
        if self.mode == 'server':
            conn, addr = self.server.accept()
            return conn

        elif self.mode == 'client':
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_address = (self.ip_address, self.port)
            num_tries = 0
            while True:
                try:
                    client.connect(client_address)
                    break
                except OSError:
                    num_tries += 1
                    if num_tries == 50:
                        raise RuntimeError('Client tried to connect too many times. Shutting down.')
                    time.sleep(0.1)
            return client

    def get_input_bytes(self, input_):
        """
        Converts the input data to bytes based on the specified protocol.

        Args:
            input_ (np.ndarray): The input data.

        Returns:
            bytes: The input data in byte format.

        Raises:
            ValueError: If the protocol is not one of 'array', 'scalar', or 'json'.
        """
        # np.array
        if self.protocol == 'array':
            input_bytes = input_.tobytes()

        # scalar
        elif self.protocol == 'scalar':
            input_str = str(input_.item())
            input_bytes = bytes(input_str, 'utf-8')

        # json
        elif self.protocol == 'json':
            input_dict = {variable.name: value for variable, value in zip(self.inputs, input_)}
            input_json = json.dumps(input_dict)
            input_bytes = bytes(input_json, 'utf-8')

        else:
            raise ValueError("'protocol' must be either array, scalar, or json")

        return input_bytes

    def get_output_data(self, output_bytes):
        """
        Converts the output bytes to data based on the specified protocol.

        Args:
            output_bytes (bytes): The output data in byte format.

        Returns:
            np.ndarray: The output data.

        Raises:
            ValueError: If the protocol is not one of 'array', 'scalar', or 'json'.
        """
        # np.array
        if self.protocol == 'array':
            output_data = np.frombuffer(output_bytes)

        # scalar
        elif self.protocol == 'scalar':
            output_str = output_bytes.decode()
            output_data = base.make_1d(float(output_str))

        # json
        elif self.protocol == 'json':
            output_str = output_bytes.decode()
            output_json = json.loads(output_str)
            print(output_json)
            output_data = base.make_1d([output_json[variable.name] for variable in self.outputs])

        else:
            raise ValueError("'protocol' must be either array, scalar, or json")

        return output_data

    def close_client(self, conn):
        """
        Closes the client connection if the mode is 'client'.

        Args:
            conn (socket.socket): The socket connection object.
        """
        if self.mode == 'client':
            conn.close()

    def update_output(self, state, input_):

        conn = self.get_connection()

        # sending data
        if len(self.inputs) > 0:
            input_bytes = self.get_input_bytes(input_)
            conn.sendall(input_bytes)
            self.close_client(conn)

        # receiving data
        else:
            output_bytes = conn.recv(128)
            self.close_client(conn)
            return self.get_output_data(output_bytes)
