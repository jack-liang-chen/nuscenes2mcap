# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: foxglove/ModelPrimitive.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from foxglove import Color_pb2 as foxglove_dot_Color__pb2
from foxglove import Pose_pb2 as foxglove_dot_Pose__pb2
from foxglove import Vector3_pb2 as foxglove_dot_Vector3__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x66oxglove/ModelPrimitive.proto\x12\x08\x66oxglove\x1a\x14\x66oxglove/Color.proto\x1a\x13\x66oxglove/Pose.proto\x1a\x16\x66oxglove/Vector3.proto\"\xb7\x01\n\x0eModelPrimitive\x12\x1c\n\x04pose\x18\x01 \x01(\x0b\x32\x0e.foxglove.Pose\x12 \n\x05scale\x18\x02 \x01(\x0b\x32\x11.foxglove.Vector3\x12\x1e\n\x05\x63olor\x18\x03 \x01(\x0b\x32\x0f.foxglove.Color\x12\x16\n\x0eoverride_color\x18\x04 \x01(\x08\x12\x0b\n\x03url\x18\x05 \x01(\t\x12\x12\n\nmedia_type\x18\x06 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x07 \x01(\x0c\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'foxglove.ModelPrimitive_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MODELPRIMITIVE._serialized_start=111
  _MODELPRIMITIVE._serialized_end=294
# @@protoc_insertion_point(module_scope)
