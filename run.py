import os
from tqdm import tqdm
import subprocess
import sys

waymo_path = "data/waymo_multi_view"
config_path = "config/psy/3dgs-waymo-{}.yaml"

prompts = {
    "104": [
        "Add a bulldozer 5m behind the black car crossing the street",
        "Add a following vehicle behind the red classic vehicle",
        "Add a parked vehicle at [-33293.25083473, 39228.75704265, -62.021820655151934]",
        "Add a traffic light at [-33291.82, 39228.66, -62.74394]",
        "Make a black car turning at the intersection to go straight",
        "Remove a black car turning at the intersection",
        "Remove all the pedestrians moving in the scene",
    ],
    "125": [
        "Remove all the pedestrians crossing the street"
    ],
    "169": [
        "Add a concrete barrier in front of construction worker with orange vest",
        "Add a new pedestrian crossing the street in front of ego",
        "Add a new vehicle going straight at the intersection",
        "Add a skid steer in front of construction worker with orange vest",
        "Make a pedestrian with red coat walking forward to walk faster"
    ],
    "584": [
        "Add a new vehicle going straight starting from next to the white car crossing the intersection",
        "Add a pedestrian sitting on the wheelchair crossing the crosswalk next to the pedestrian standing on the left side",
        "Add a pedestrian sitting on the wheelchair next to the pedestrian standing on the left side",
        "Add a traffic cone at the left crosswalk at [-414.4302, 15551.24, -21.09849]",
        "Make the white car moving left to right to stop at the intersection", 
        "Remove the black bus across the street"
    ],
    "776": [
        "Add a following vehicle behind a sedan crossing the intersection moving right to left",
        "Add a illegally parked car in front of a sedan crossing the intersection moving right to left",
        "Make a sedan crossing the intersection moving right to left in front of ego to turn right",
        "Make a vehicle moving forward from the opposite lane toward ego to slow down and stop",
        "Remove a vehicle turning right at the intersection",
        "Remove all the moving vehicles on the road"
    ],
    "448": [
        "Add a stop sign next to the bus ahead",
        "Replace a bus ahead of ego into a uhaul truck",
        "Replace a yellow taxi stopped at the intserction on the left into white lamborghini"
    ],
    "965": [
        "Add a jaywalking pedestrian walking from (x,y,z) to (x,y,z)",
        "Add a jaywalking pedestrian walking from (x,y,z) to (x,y,z)_with_refinement"
    ]
}

def main():
    for scene in tqdm(os.listdir(waymo_path)):
        if not os.path.isdir(os.path.join(waymo_path, scene)):
            continue
        scene_id = scene.split('-')[1][:3]
        scene_path = os.path.join(waymo_path, scene)
        scene_config_path = config_path.format(scene_id)

        try:
            if scene_id in prompts:
                for idx, prompt in enumerate(prompts[scene_id]):
                    command = [
                        "python", "main.py",
                        "-y", scene_config_path,
                        "-p", prompt,
                        "-s", f"{scene_id}_{idx}"
                    ]
                    print(" ".join(command))
                    # Run with Ctrl+C passthrough
                    try:
                        subprocess.run(command, check=True)
                    except subprocess.CalledProcessError as e:
                        # Child exited with non-zero status: log it and continue to next prompt
                        print(f"[skip] scene {scene_id}, prompt '{idx}' failed with code {e.returncode}")
                        continue
        except KeyboardInterrupt:
            print("\nStopping all runs by user request (Ctrl+C).")
            sys.exit(0)
            # for idx, prompt in enumerate(prompts[scene_id]):
            #     command = f"python main.py -y {scene_config_path} -p {prompt} -s {scene_id}_{idx}"
            #     print(command)
            #     try:
            #         os.system(command)
            #     except KeyboardInterrupt:
            #         print("Stopping...")
            #         sys.exit(0)
            #     except Exception as e:
            #         print(f"Error processing scene {scene} with prompt {prompt}: {e}")


if __name__ == "__main__":
    main()