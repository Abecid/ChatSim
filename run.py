import os
from tqdm import tqdm
import subprocess
import sys

waymo_path = "data/waymo_multi_view"
config_path = "config/psy/3dgs-waymo-{}.yaml"

prompts = {
    "104": [
        "Add a bulldozer 5m behind the intersection",
        "Add a red vehicle turning at the intersection",
        "Add a parked vehicle at [20, -10, 0]",
        "Add a traffic cone at [20, -10, 0]",
        "Create a bmw turning at the intersection",
        "Create porsche 911 going straight at the intersection",
        # "Remove all the pedestrians moving in the scene",
    ],
    "125": [
        # "Remove all the pedestrians crossing the street"
    ],
    "169": [
        "Add a sign fence in the intersection",
        "Add a new Tesla roadster crossing the street in front of ego",
        "Add an audi going straight at the intersection",
        "Add a loader truck in front of the intersection",
        "Make a benz g go forward fast"
    ],
    "584": [
        "Add a cadillac going straight starting from ego crossing the intersection",
        "Add an audi crossing the crosswalk next to the pedestrian standing on the left side",
        "Add a excavator next to the pedestrian standing on the left side",
        "Add a traffic cone at the left crosswalk at [20, 10, 0]",
        "Make a chevrolet moving left to right to stop at the intersection", 
        # "Remove the black bus across the street"
    ],
    "776": [
        "Add a m1a2 tank behind a sedan crossing the intersection moving right to left",
        "Add an illegally parked benz s in front of a sedan crossing the intersection moving right to left",
        "Make an audi turn right at the intersection",
        "Make a ferrari coming towards me from the opposite lane to slow down and stop",
        "Create a dodge srt turning right at the intersection",
        # "Remove all the moving vehicles on the road"
    ],
    "448": [
        "Add a traffic cone next to the bus ahead",
        "Replace a bus ahead of ego into a lamborghini",
        "Replace a yellow taxi stopped at the intserction on the left into lamborghini"
    ],
    "965": [
        "Add a land rover range going from (10,0,0) to (30,-20,0)",
        "Add a ferrari going from (10,-10,0) to (0,0,0)_with_refinement"
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