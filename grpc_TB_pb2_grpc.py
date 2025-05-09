# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import grpc_TB_pb2 as grpc__TB__pb2


class gRPCTBStub(object):
    """This service interacts with a tag bus-based system.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListChannels = channel.unary_unary(
                '/gRPCTB.gRPCTB/ListChannels',
                request_serializer=grpc__TB__pb2.ChannelRequest.SerializeToString,
                response_deserializer=grpc__TB__pb2.ChannelList.FromString,
                )
        self.GetDouble = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetDouble',
                request_serializer=grpc__TB__pb2.DoubleName.SerializeToString,
                response_deserializer=grpc__TB__pb2.DoubleValue.FromString,
                )
        self.GetFloat = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetFloat',
                request_serializer=grpc__TB__pb2.FloatName.SerializeToString,
                response_deserializer=grpc__TB__pb2.FloatValue.FromString,
                )
        self.GetInt64 = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetInt64',
                request_serializer=grpc__TB__pb2.Int64Name.SerializeToString,
                response_deserializer=grpc__TB__pb2.Int64Value.FromString,
                )
        self.GetInt32 = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetInt32',
                request_serializer=grpc__TB__pb2.Int32Name.SerializeToString,
                response_deserializer=grpc__TB__pb2.Int32Value.FromString,
                )
        self.GetUint64 = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetUint64',
                request_serializer=grpc__TB__pb2.Uint64Name.SerializeToString,
                response_deserializer=grpc__TB__pb2.Uint64Value.FromString,
                )
        self.GetUint32 = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetUint32',
                request_serializer=grpc__TB__pb2.Uint32Name.SerializeToString,
                response_deserializer=grpc__TB__pb2.Uint32Value.FromString,
                )
        self.GetBool = channel.unary_unary(
                '/gRPCTB.gRPCTB/GetBool',
                request_serializer=grpc__TB__pb2.BoolValue.SerializeToString,
                response_deserializer=grpc__TB__pb2.BoolValue.FromString,
                )
        self.SetDouble = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetDouble',
                request_serializer=grpc__TB__pb2.DoubleValue.SerializeToString,
                response_deserializer=grpc__TB__pb2.DoubleReply.FromString,
                )
        self.SetFloat = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetFloat',
                request_serializer=grpc__TB__pb2.FloatValue.SerializeToString,
                response_deserializer=grpc__TB__pb2.FloatReply.FromString,
                )
        self.SetInt64 = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetInt64',
                request_serializer=grpc__TB__pb2.Int64Value.SerializeToString,
                response_deserializer=grpc__TB__pb2.Int64Reply.FromString,
                )
        self.SetInt32 = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetInt32',
                request_serializer=grpc__TB__pb2.Int32Value.SerializeToString,
                response_deserializer=grpc__TB__pb2.Int32Reply.FromString,
                )
        self.SetUint64 = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetUint64',
                request_serializer=grpc__TB__pb2.Uint64Value.SerializeToString,
                response_deserializer=grpc__TB__pb2.Uint64Reply.FromString,
                )
        self.SetUint32 = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetUint32',
                request_serializer=grpc__TB__pb2.Uint32Value.SerializeToString,
                response_deserializer=grpc__TB__pb2.Uint32Reply.FromString,
                )
        self.SetBool = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetBool',
                request_serializer=grpc__TB__pb2.BoolValue.SerializeToString,
                response_deserializer=grpc__TB__pb2.BoolReply.FromString,
                )
        self.ListAxes = channel.unary_unary(
                '/gRPCTB.gRPCTB/ListAxes',
                request_serializer=grpc__TB__pb2.AxisRequest.SerializeToString,
                response_deserializer=grpc__TB__pb2.AxisList.FromString,
                )
        self.SetAxisPosition = channel.unary_unary(
                '/gRPCTB.gRPCTB/SetAxisPosition',
                request_serializer=grpc__TB__pb2.AxisValue.SerializeToString,
                response_deserializer=grpc__TB__pb2.AxisReply.FromString,
                )
        self.EnableAxis = channel.unary_unary(
                '/gRPCTB.gRPCTB/EnableAxis',
                request_serializer=grpc__TB__pb2.AxisList.SerializeToString,
                response_deserializer=grpc__TB__pb2.AxisReply.FromString,
                )
        self.DisableAxis = channel.unary_unary(
                '/gRPCTB.gRPCTB/DisableAxis',
                request_serializer=grpc__TB__pb2.AxisList.SerializeToString,
                response_deserializer=grpc__TB__pb2.AxisReply.FromString,
                )
        self.ClearFaults = channel.unary_unary(
                '/gRPCTB.gRPCTB/ClearFaults',
                request_serializer=grpc__TB__pb2.AxisList.SerializeToString,
                response_deserializer=grpc__TB__pb2.AxisReply.FromString,
                )


class gRPCTBServicer(object):
    """This service interacts with a tag bus-based system.
    """

    def ListChannels(self, request, context):
        """Returns a list of channels
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDouble(self, request, context):
        """get channel values
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFloat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetInt64(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetInt32(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetUint64(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetUint32(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetBool(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetDouble(self, request, context):
        """set channel values
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetFloat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetInt64(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetInt32(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetUint64(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetUint32(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetBool(self, request, context):
        """yes... this is a typo, but with the code that would have to be regenerated, I don't know if it's worth it to fix at this point.  Let me know if you want it changed.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListAxes(self, request, context):
        """This could be a separate service, but we'll make it the same service for the sake of the code behind it.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAxisPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def EnableAxis(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DisableAxis(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ClearFaults(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_gRPCTBServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ListChannels': grpc.unary_unary_rpc_method_handler(
                    servicer.ListChannels,
                    request_deserializer=grpc__TB__pb2.ChannelRequest.FromString,
                    response_serializer=grpc__TB__pb2.ChannelList.SerializeToString,
            ),
            'GetDouble': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDouble,
                    request_deserializer=grpc__TB__pb2.DoubleName.FromString,
                    response_serializer=grpc__TB__pb2.DoubleValue.SerializeToString,
            ),
            'GetFloat': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFloat,
                    request_deserializer=grpc__TB__pb2.FloatName.FromString,
                    response_serializer=grpc__TB__pb2.FloatValue.SerializeToString,
            ),
            'GetInt64': grpc.unary_unary_rpc_method_handler(
                    servicer.GetInt64,
                    request_deserializer=grpc__TB__pb2.Int64Name.FromString,
                    response_serializer=grpc__TB__pb2.Int64Value.SerializeToString,
            ),
            'GetInt32': grpc.unary_unary_rpc_method_handler(
                    servicer.GetInt32,
                    request_deserializer=grpc__TB__pb2.Int32Name.FromString,
                    response_serializer=grpc__TB__pb2.Int32Value.SerializeToString,
            ),
            'GetUint64': grpc.unary_unary_rpc_method_handler(
                    servicer.GetUint64,
                    request_deserializer=grpc__TB__pb2.Uint64Name.FromString,
                    response_serializer=grpc__TB__pb2.Uint64Value.SerializeToString,
            ),
            'GetUint32': grpc.unary_unary_rpc_method_handler(
                    servicer.GetUint32,
                    request_deserializer=grpc__TB__pb2.Uint32Name.FromString,
                    response_serializer=grpc__TB__pb2.Uint32Value.SerializeToString,
            ),
            'GetBool': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBool,
                    request_deserializer=grpc__TB__pb2.BoolValue.FromString,
                    response_serializer=grpc__TB__pb2.BoolValue.SerializeToString,
            ),
            'SetDouble': grpc.unary_unary_rpc_method_handler(
                    servicer.SetDouble,
                    request_deserializer=grpc__TB__pb2.DoubleValue.FromString,
                    response_serializer=grpc__TB__pb2.DoubleReply.SerializeToString,
            ),
            'SetFloat': grpc.unary_unary_rpc_method_handler(
                    servicer.SetFloat,
                    request_deserializer=grpc__TB__pb2.FloatValue.FromString,
                    response_serializer=grpc__TB__pb2.FloatReply.SerializeToString,
            ),
            'SetInt64': grpc.unary_unary_rpc_method_handler(
                    servicer.SetInt64,
                    request_deserializer=grpc__TB__pb2.Int64Value.FromString,
                    response_serializer=grpc__TB__pb2.Int64Reply.SerializeToString,
            ),
            'SetInt32': grpc.unary_unary_rpc_method_handler(
                    servicer.SetInt32,
                    request_deserializer=grpc__TB__pb2.Int32Value.FromString,
                    response_serializer=grpc__TB__pb2.Int32Reply.SerializeToString,
            ),
            'SetUint64': grpc.unary_unary_rpc_method_handler(
                    servicer.SetUint64,
                    request_deserializer=grpc__TB__pb2.Uint64Value.FromString,
                    response_serializer=grpc__TB__pb2.Uint64Reply.SerializeToString,
            ),
            'SetUint32': grpc.unary_unary_rpc_method_handler(
                    servicer.SetUint32,
                    request_deserializer=grpc__TB__pb2.Uint32Value.FromString,
                    response_serializer=grpc__TB__pb2.Uint32Reply.SerializeToString,
            ),
            'SetBool': grpc.unary_unary_rpc_method_handler(
                    servicer.SetBool,
                    request_deserializer=grpc__TB__pb2.BoolValue.FromString,
                    response_serializer=grpc__TB__pb2.BoolReply.SerializeToString,
            ),
            'ListAxes': grpc.unary_unary_rpc_method_handler(
                    servicer.ListAxes,
                    request_deserializer=grpc__TB__pb2.AxisRequest.FromString,
                    response_serializer=grpc__TB__pb2.AxisList.SerializeToString,
            ),
            'SetAxisPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.SetAxisPosition,
                    request_deserializer=grpc__TB__pb2.AxisValue.FromString,
                    response_serializer=grpc__TB__pb2.AxisReply.SerializeToString,
            ),
            'EnableAxis': grpc.unary_unary_rpc_method_handler(
                    servicer.EnableAxis,
                    request_deserializer=grpc__TB__pb2.AxisList.FromString,
                    response_serializer=grpc__TB__pb2.AxisReply.SerializeToString,
            ),
            'DisableAxis': grpc.unary_unary_rpc_method_handler(
                    servicer.DisableAxis,
                    request_deserializer=grpc__TB__pb2.AxisList.FromString,
                    response_serializer=grpc__TB__pb2.AxisReply.SerializeToString,
            ),
            'ClearFaults': grpc.unary_unary_rpc_method_handler(
                    servicer.ClearFaults,
                    request_deserializer=grpc__TB__pb2.AxisList.FromString,
                    response_serializer=grpc__TB__pb2.AxisReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gRPCTB.gRPCTB', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class gRPCTB(object):
    """This service interacts with a tag bus-based system.
    """

    @staticmethod
    def ListChannels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/ListChannels',
            grpc__TB__pb2.ChannelRequest.SerializeToString,
            grpc__TB__pb2.ChannelList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDouble(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetDouble',
            grpc__TB__pb2.DoubleName.SerializeToString,
            grpc__TB__pb2.DoubleValue.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFloat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetFloat',
            grpc__TB__pb2.FloatName.SerializeToString,
            grpc__TB__pb2.FloatValue.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetInt64(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetInt64',
            grpc__TB__pb2.Int64Name.SerializeToString,
            grpc__TB__pb2.Int64Value.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetInt32(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetInt32',
            grpc__TB__pb2.Int32Name.SerializeToString,
            grpc__TB__pb2.Int32Value.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetUint64(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetUint64',
            grpc__TB__pb2.Uint64Name.SerializeToString,
            grpc__TB__pb2.Uint64Value.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetUint32(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetUint32',
            grpc__TB__pb2.Uint32Name.SerializeToString,
            grpc__TB__pb2.Uint32Value.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetBool(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/GetBool',
            grpc__TB__pb2.BoolValue.SerializeToString,
            grpc__TB__pb2.BoolValue.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetDouble(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetDouble',
            grpc__TB__pb2.DoubleValue.SerializeToString,
            grpc__TB__pb2.DoubleReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetFloat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetFloat',
            grpc__TB__pb2.FloatValue.SerializeToString,
            grpc__TB__pb2.FloatReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetInt64(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetInt64',
            grpc__TB__pb2.Int64Value.SerializeToString,
            grpc__TB__pb2.Int64Reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetInt32(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetInt32',
            grpc__TB__pb2.Int32Value.SerializeToString,
            grpc__TB__pb2.Int32Reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetUint64(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetUint64',
            grpc__TB__pb2.Uint64Value.SerializeToString,
            grpc__TB__pb2.Uint64Reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetUint32(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetUint32',
            grpc__TB__pb2.Uint32Value.SerializeToString,
            grpc__TB__pb2.Uint32Reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetBool(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetBool',
            grpc__TB__pb2.BoolValue.SerializeToString,
            grpc__TB__pb2.BoolReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListAxes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/ListAxes',
            grpc__TB__pb2.AxisRequest.SerializeToString,
            grpc__TB__pb2.AxisList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAxisPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/SetAxisPosition',
            grpc__TB__pb2.AxisValue.SerializeToString,
            grpc__TB__pb2.AxisReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def EnableAxis(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/EnableAxis',
            grpc__TB__pb2.AxisList.SerializeToString,
            grpc__TB__pb2.AxisReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DisableAxis(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/DisableAxis',
            grpc__TB__pb2.AxisList.SerializeToString,
            grpc__TB__pb2.AxisReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClearFaults(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gRPCTB.gRPCTB/ClearFaults',
            grpc__TB__pb2.AxisList.SerializeToString,
            grpc__TB__pb2.AxisReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
