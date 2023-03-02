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
    
    filepath.parent.mkdir(parent=True, exist_ok=True)
    
    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)

        
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