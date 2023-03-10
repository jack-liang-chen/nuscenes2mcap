"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import foxglove.PackedElementField_pb2
import foxglove.Pose_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import google.protobuf.timestamp_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class PointCloud(google.protobuf.message.Message):
    """A collection of N-dimensional points, which may contain additional fields with information like normals, intensity, etc."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TIMESTAMP_FIELD_NUMBER: builtins.int
    FRAME_ID_FIELD_NUMBER: builtins.int
    POSE_FIELD_NUMBER: builtins.int
    POINT_STRIDE_FIELD_NUMBER: builtins.int
    FIELDS_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Timestamp of point cloud"""
        pass
    frame_id: typing.Text
    """Frame of reference"""

    @property
    def pose(self) -> foxglove.Pose_pb2.Pose:
        """The origin of the point cloud relative to the frame of reference"""
        pass
    point_stride: builtins.int
    """Number of bytes between points in the `data`"""

    @property
    def fields(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[foxglove.PackedElementField_pb2.PackedElementField]:
        """Fields in the `data`"""
        pass
    data: builtins.bytes
    """Point data, interpreted using `fields`"""

    def __init__(self,
        *,
        timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        frame_id: typing.Text = ...,
        pose: typing.Optional[foxglove.Pose_pb2.Pose] = ...,
        point_stride: builtins.int = ...,
        fields: typing.Optional[typing.Iterable[foxglove.PackedElementField_pb2.PackedElementField]] = ...,
        data: builtins.bytes = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["pose",b"pose","timestamp",b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["data",b"data","fields",b"fields","frame_id",b"frame_id","point_stride",b"point_stride","pose",b"pose","timestamp",b"timestamp"]) -> None: ...
global___PointCloud = PointCloud
