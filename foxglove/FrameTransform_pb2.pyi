"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import foxglove.Quaternion_pb2
import foxglove.Vector3_pb2
import google.protobuf.descriptor
import google.protobuf.message
import google.protobuf.timestamp_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class FrameTransform(google.protobuf.message.Message):
    """A transform between two reference frames in 3D space"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TIMESTAMP_FIELD_NUMBER: builtins.int
    PARENT_FRAME_ID_FIELD_NUMBER: builtins.int
    CHILD_FRAME_ID_FIELD_NUMBER: builtins.int
    TRANSLATION_FIELD_NUMBER: builtins.int
    ROTATION_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Timestamp of transform"""
        pass
    parent_frame_id: typing.Text
    """Name of the parent frame"""

    child_frame_id: typing.Text
    """Name of the child frame"""

    @property
    def translation(self) -> foxglove.Vector3_pb2.Vector3:
        """Translation component of the transform"""
        pass
    @property
    def rotation(self) -> foxglove.Quaternion_pb2.Quaternion:
        """Rotation component of the transform"""
        pass
    def __init__(self,
        *,
        timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        parent_frame_id: typing.Text = ...,
        child_frame_id: typing.Text = ...,
        translation: typing.Optional[foxglove.Vector3_pb2.Vector3] = ...,
        rotation: typing.Optional[foxglove.Quaternion_pb2.Quaternion] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["rotation",b"rotation","timestamp",b"timestamp","translation",b"translation"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["child_frame_id",b"child_frame_id","parent_frame_id",b"parent_frame_id","rotation",b"rotation","timestamp",b"timestamp","translation",b"translation"]) -> None: ...
global___FrameTransform = FrameTransform
