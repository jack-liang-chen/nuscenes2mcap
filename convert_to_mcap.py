import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rospy

from mcap.writer import Writer, CompressionType
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from tqdm import tqdm

from foxglove.CameraCalibration_pb2 import CameraCalibration
from foxglove.CompressedImage_pb2 import CompressedImage
from foxglove.FrameTransform_pb2 import FrameTransform
from foxglove.Grid_pb2 import Grid
from foxglove.ImageAnnotations_pb2 import ImageAnnotations
from foxglove.LinePrimitive_pb2 import LinePrimitive
from foxglove.LocationFix_pb2 import LocationFix
from foxglove.PackedElementField_pb2 import PackedElementField
from foxglove.PointCloud_pb2 import PointCloud
from foxglove.PoseInFrame_pb2 import PoseInFrame
from foxglove.PointsAnnotation_pb2 import PointsAnnotation
from foxglove.Quaternion_pb2 import Quaternion as foxglove_Quaternion
from foxglove.SceneUpdate_pb2 import SceneUpdate
from foxglove.Vector3_pb2 import Vector3
from ProtobufWriter import ProtobufWriter
from RosmsgWriter import RosmsgWriter

with open(Path(__file__).parent / "turbomap.json") as f:
    TURBOMAP_DATA = np.array(json.load(f))


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md#imu
IMU_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "linear_accel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "q": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "rotation_rate": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
    },
}

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md#pose
ODOM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "accel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "orientation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "pos": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "rotation_rate": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "vel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
    },
}


def load_bitmap(dataroot: str, map_name: str, layer_name: str) -> np.array:
    # Load bitmap.
    if layer_name == "basemap":
        map_path = os.path.join(dataroot, "maps", "basemap", map_name + ".png")
    else: 
        raise Exception("Error: Invalid bitmap layer: %s" % layer_name)
        
    # Convert to numpy.
    if os.path.exists(map_path):
        image = np.array(Image.open(map_path).convert("L"))
    else:
        raise Exception("Error: Cannot find %s %s! Please make sure that the map is correctly installed." %(layer_name, map_path))
    
    # Invert semantic prior colors.

    return image


def get_num_sample_data(nusc: NuScenes, scene):
    num_sample_data = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sample_data = nusc.get("sample_data", sample_token)
        while sample_data is not None:
            num_sample_data += 1
            sample_data = nusc.get("sample_data", sample_data["next"]) if sample_data["next"] != "" else None
    return num_sample_data
 

def get_time(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["timestamp"], 1_000_000)
    t.nsecs = msecs * 1000

    return t


def get_utime(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["utime"], 1_000_000)
    t.nsecs = msecs * 1000

    return t


def get_imu_msg(imu_data):
    timestamp = get_utime(imu_data)
    
    msg = {
        "linear_accel": {
            "x": imu_data["linear_accel"][0],
            "y": imu_data["linear_accel"][1],
            "z": imu_data["linear_accel"][2],
        },
        "q": {
            "w": imu_data["q"][0],
            "x": imu_data["q"][1],
            "y": imu_data["q"][2],
            "z": imu_data["q"][3],
        },
        "rotation_rate": {
            "x": imu_data["rotation_rate"][0],
            "y": imu_data["rotation_rate"][1],
            "z": imu_data["rotation_rate"][2],
        },
    }
    
    return (timestamp, "/imu", json.dumps(msg).encode())


def get_odom_msg(pose_data):
    timestamp = get_utime(pose_data)
    
    msg = {
        "accel": {
            "x": pose_data["accel"][0],
            "y": pose_data["accel"][1],
            "z": pose_data["accel"][2],
        },
        "orientation": {
            "w": pose_data["orientation"][0],
            "x": pose_data["orientation"][1],
            "y": pose_data["orientation"][2],
            "z": pose_data["orientation"][3],
        },
        "pos": {
            "x": pose_data["pos"][0],
            "y": pose_data["pos"][1],
            "z": pose_data["pos"][2],
        },
        "rotation_rate": {
            "x": pose_data["rotation_rate"][0],
            "y": pose_data["rotation_rate"][1],
            "z": pose_data["rotation_rate"][2],
        },
        "vel": {
            "x": pose_data["vel"][0],
            "y": pose_data["vel"][1],
            "z": pose_data["vel"][2],
        },
    }
    
    return (timestamp, "/odom", json.dumps(msg).encode())


def get_basic_can_msg(name, diag_data):
    pass    
 

def scene_bounding_box(nusc, scene, nusc_map, padding=75.0):
    box = [np.inf, np.inf, -np.inf, -np.inf]
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        x, y = ego_pose["translation"][:2]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None
    box[0] = max(box[0] - padding, 0.0)
    box[1] = max(box[1] - padding, 0.0)
    box[2] = min(box[2] + padding, nusc_map.canvas_edge[0]) - box[0]
    box[3] = min(box[3] + padding, nusc_map.canvas_edge[1]) - box[1]
    return box
    

def get_scene_map(nusc, scene, nusc_map, image, stamp):
    x, y, w, h = scene_bounding_box(nusc, scene, nusc_map)
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    img = np.flipud(image)[img_y : img_y + img_h, img_x : img_x + img_w]
    
    # img values are 0-255
    # convert to a color scale, 0=white and 255=black, in packed RGBA format: 0xFFFFFF00 to 0x00000000
    img = (255 - img) * 0x01010100
    # set alpha to 0xFF for all cells except those that are completely black
    img[img != 0x00000000] |= 0x000000FF

    msg = Grid()
    msg.timestamp.FromNanoseconds(stamp.to_nsec())
    msg.frame_id = "map"
    msg.cell_size.x = 0.1
    msg.cell_size.y = 0.1
    msg.column_count = img_w
    msg.row_stride = img_w * 4
    msg.cell_stride = 4
    msg.fields.add(name="alpha", offset=0, type=PackedElementField.UINT8)
    msg.fields.add(name="blue", offset=1, type=PackedElementField.UINT8)
    msg.fields.add(name="green", offset=2, type=PackedElementField.UINT8)
    msg.fields.add(name="red", offset=3, type=PackedElementField.UINT8)
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1
    msg.data = img.astype("<u4").tobytes()

    return msg


def rectContains(rect, point):
    a, b, c, d = rect
    x, y = point[:2]
    return a <= x < a + c and b <= y < b + d


def get_centerline_markers(nusc, scene, nusc_map, stamp):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained_pose_lists = []
    for pose_list in pose_lists:
        new_pose_list = []
        for pose in pose_list:
            if rectContains(bbox, pose):
                new_pose_list.append(pose)
        if len(new_pose_list) > 1:
            contained_pose_lists.append(new_pose_list)

    scene_update = SceneUpdate()
    for i, pose_list in enumerate(contained_pose_lists):
        entity = scene_update.entities.add()
        entity.frame_id = "map"
        entity.timestamp.FromNanoseconds(stamp.to_nsec())
        entity.id = f"{i}"
        entity.frame_locked = True
        line = entity.lines.add()
        line.type = LinePrimitive.Type.LINE_STRIP
        line.thickness = 0.1
        line.color.r = 51.0 / 255.0
        line.color.g = 160.0 / 255.0
        line.color.b = 44.0 / 255.0
        line.color.a = 1.0
        line.pose.orientation.w = 1.0
        for pose in pose_list:
            line.points.add(x=pose[0], y=pose[1], z=0)

    return scene_update 


def write_scene_to_mcap(nusc: NuScenes, nusc_can: NuScenesCanBus, scene, filepath):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    print(f'Loading map "{location}"')
    data_path = Path(nusc.dataroot)
    nusc_map = NuScenesMap(dataroot=data_path, map_name=location)
    print(f'Loading bitmap "{nusc_map.map_name}"')
    image = load_bitmap(nusc_map.dataroot, nusc_map.map_name, "basemap")
    print(f"Loaded {image.shape} bitmap")
    print(f"vehicle is {log['vehicle']}")
    
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene_name} Sample Data", leave=False) 
    
    can_parsers = [
        [nusc_can.get_messages(scene_name, "ms_imu"), 0, get_imu_msg],
        [nusc_can.get_messages(scene_name, "pose"), 0, get_odom_msg],
        [
            nusc_can.get_messages(scene_name, "steeranglefeedback"),
            0,
            lambda x: get_basic_can_msg("Steering Angle", x),
        ],
        [
            nusc_can.get_messages(scene_name, "vehicle_monitor"),
            0,
            lambda x: get_basic_can_msg("Vehicle Monitor", x),
        ],
        [
            nusc_can.get_messages(scene_name, "zoesensors"),
            0,
            lambda x: get_basic_can_msg("Zoe Sensors", x),
        ],
        [
            nusc_can.get_messages(scene_name, "zoe_veh_info"),
            0,
            lambda x: get_basic_can_msg("Zoe Vehicle Info", x),
        ],
    ]
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)

        imu_schema_id = writer.register_schema(name="IMU", encoding="jsonschema", data=json.dumps(IMU_JSON_SCHEMA).encode())
        imu_channel_id = writer.register_channel(topic="/imu", message_encoding="json", schema_id=imu_schema_id)

        odom_schema_id = writer.register_schema(name="Pose", encoding="jsonschema", data=json.dumps(ODOM_JSON_SCHEMA).encode())
        odom_channel_id = writer.register_channel(topic="/odom", message_encoding="json", schema_id=odom_schema_id)

        protobuf_writer = ProtobufWriter(writer)
        rosmsg_writer = RosmsgWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")
        
        writer.add_metadata(
            "scene-info",
            {
                "description": scene["description"],
                "name": scene["name"],
                "location": location,
                "vehicle": log["vehicle"],
                "date_captured": log["date_captured"],
            },
        )
        
        stamp = get_time(
            nusc.get(
                "ego_pose",
                nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])["ego_pose_token"],
            )
        )
        map_msg = get_scene_map(nusc, scene, nusc_map, image, stamp)
        centerlines_msg = get_centerline_markers(nusc, scene, nusc_map, stamp)
        protobuf_writer.write_message("/map", map_msg, stamp.to_nsec())
        protobuf_writer.write_message("/semantic_map", centerlines_msg, stamp.to_nsec())
        
        
        
        while cur_sample is not None:
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            stamp = get_time(ego_pose)
            
            # write CAN message to /pose, /odom, and /diagnostics
            
            # publish /tf
            
            # /driveable_area occupancy grid
            
            # iterate sensors
            
            # publish /gps
            
            # publish /markers/annotations
            
            # publish /markers/car
            
            # collect all sensor frames after this sample but before the next sample
            
            # sort and publish the non-keyframe sensor msgs
            
            # move to the next sample
        
        pbar.close()
        writer.finish()
        print(f"Finished writing {filepath}")
    

def convert_all(
    output_dit: Path,
    name: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus,
    selected_scenes,
):
    nusc.list_scenes()
    for scene in nusc.scene:
        scene_name = scene["name"]
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        mcap_name = f"Nuscenes-{name}-{scene_name}.mcap"
        write_scene_to_mcap(nusc, nusc_can, scene, output_dit / mcap_name)     


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--data-dir",
        "-d",
        default=script_dir / "data",
        help="path to nuscenes data directory",
    )
    parser.add_argument(
        "--dataset-name",
        "-n",
        default=["v1.0-mini"],
        nargs="+",
        help="dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=script_dir / "output",
        help="path to write MCAP files into",
    )
    parser.add_argument( "--scene", "-s", nargs="*", help="specific scene(s) to write" )
    parser.add_argument( "--list-only", action="store_true", help="lists the scenes and exists" )
    
    args = parser.parse_args()
    
    nusc_can = NuScenesCanBus(dataroot=str(args.data_dir))
    
    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return
        convert_all(args.output_dir, name, nusc, nusc_can, args.scene)            


if __name__ == "__main__":
    main()