"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import foxglove.CircleAnnotation_pb2
import foxglove.PointsAnnotation_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class ImageAnnotations(google.protobuf.message.Message):
    """Array of annotations for a 2D image"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    CIRCLES_FIELD_NUMBER: builtins.int
    POINTS_FIELD_NUMBER: builtins.int
    @property
    def circles(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[foxglove.CircleAnnotation_pb2.CircleAnnotation]:
        """Circle annotations"""
        pass
    @property
    def points(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[foxglove.PointsAnnotation_pb2.PointsAnnotation]:
        """Points annotations"""
        pass
    def __init__(self,
        *,
        circles: typing.Optional[typing.Iterable[foxglove.CircleAnnotation_pb2.CircleAnnotation]] = ...,
        points: typing.Optional[typing.Iterable[foxglove.PointsAnnotation_pb2.PointsAnnotation]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["circles",b"circles","points",b"points"]) -> None: ...
global___ImageAnnotations = ImageAnnotations
