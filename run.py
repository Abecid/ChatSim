import os
from tqdm import tqdm
import subprocess
import sys

waymo_path = "data/waymo_multi_view"
config_path = "config/psy/3dgs-waymo-{}.yaml"

prompts = {
    "104": [
        "Make the red classic car turn right at the intersection",
        "Make the red classic car go straight at the intersection",
        "Make the red classic car stop at the intersection",
        "Add a following vehicle 5m behind the red classic vehicle",
        "Add a vehicle 5m ahead of the red classic vehicle",
        "Add a stopped vehicle 5m ahead of the red classic vehicle",
        "Add a traffic cone 5m behind the black car crossing the street",
        "Remove a red classic vehicle",
        "Remove a pedestrian with red backpack",
        "Remove all vehicles",
        "Add a pedestrian crossing the street",
        "Make a pedestrian with red backpack walking forward to walk faster",
        "Make a pedestrian with red backpack walking forward to walk slower",
        "Make a pedestrian with pink jacket to walk faster at the crosswalk",
        "Make a pedestrian with pink jacket to walk slower at the crosswalk"
    ],
    "125": [
        "Add a pedestrian jaywalking in front of ego vehicle",
        "Add a pedestrian crossing the street on the right side of ego vehicle"
    ],
    "169": [
        "Add a new vehicle turning left at the intersection",
        "Add a new vehicle stopped at the intersection",
        "Add a new vehicle following the black SUV at the intersection",
        "Remove a pedestrian with red coat",
        "Remove a constructino worker",
        "Remove a benz stopped at the intersection",
        "Add a pedestrian crossing the street in front of ego vehicle",
        "Add a pedestrian crossing the street on the left side of ego vehicle",
        "Make a pedestrian with red coat walking forward to stop",
        "Make a construction worker with orange vest to cross the street faster",
        "Make a construction worker with orange vest to cross the street slower"
    ],
    "584": [
        "Make the white car moving left to right turn left at the intersection",
        "Make the white car moving left to right accelerate at the intersection",
        "Make the white car moving left to right turn right at the intersection",
        "Add a new vehicle turning left at the intersection starting at [-417.7167, 15544.44, -20.79474]",
        "Add a new vehicle turning right starting at [-417.7167, 15544.44, -20.79474]",
        "Add a pedestrian crossing the street in front of ego vehicle",
        "Make a pedestrian with beige coat walk slower at the crosswalk",
        "Make a pedestrian with beige coat walk faster at the crosswalk",
        "Make a pedestrian with beige coat stop at the crosswalk"
    ],
    "776": [
        "Make a sedan crossing the intersection moving right to left in front of ego turn right",
        "Make a sedan crossing the intersection moving right to left in front of ego turn left",
        "Make a sedan crossing the intersection moving right to left in front of ego stop",
        "Make a sedan crossing the intersection moving right to left in front of ego accelerate",
        "Add a following vehicle 5m behind a sedan crossing the intersection moving right to left",
        "Add a concrete barrier in front of a sedan crossing the intersection moving right to left",
        "Add a traffic cone in front of a sedan crossing the intersection moving right to left"
    ],
    "448": [
        "Add a traffic light next to the bus ahead",
        "Remove a yellow taxi stopped at the intersection",
        "Remove a bus ahead",
        "Remove all the vehicles",
        "Add a pedestrian crossing the street in front of ego vehicle",
        "Add a pedestrian walking on the sidewalk at the right side of ego vehicle"
    ],
    "965": [
        "Add a road block in front of grey car going straight",
        "Add a traffic cone in front of grey car going straight"
    ]
}

prepend_prompt = "Remove all cars in the scene and"

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
                    prompt = f"{prepend_prompt} {prompt}"
                    command = [
                        "python", "main.py",
                        "-y", scene_config_path,
                        "-p", prompt,
                        "-s", f"idx_{idx}"
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