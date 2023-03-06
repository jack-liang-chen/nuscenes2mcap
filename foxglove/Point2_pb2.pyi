"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Point2(google.protobuf.message.Message):
    """A point representing a position in 2D space"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    X_FIELD_NUMBER: builtins.int
    Y_FIELD_NUMBER: builtins.int
    x: builtins.float
    """x coordinate position"""

    y: builtins.float
    """y coordinate position"""

    def __init__(self,
        *,
        x: builtins.float = ...,
        y: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["x",b"x","y",b"y"]) -> None: ...
global___Point2 = Point2
