#!/bin/bash
DATA_DIR="../../../data/waymo_multi_view"

skips=("segment-4487677815262010875_4940_000_4960_000_with_camera_labels")

for SCENE_NAME in $(ls $DATA_DIR); do
    if [[ " ${skips[@]} " =~ " $SCENE_NAME " ]]; then
        echo "Skipping $SCENE_NAME"
        continue
    fi

    echo "Training on $SCENE_NAME..."
    python train.py --config configs/chatsim/original.yaml source_path=~/workspace/adam/ChatSim/data/waymo_multi_view/${SCENE_NAME}/colmap/sparse_undistorted model_path=output/${SCENE_NAME}

    echo "Rendering $SCENE_NAME..."
    python render.py -m output/${SCENE_NAME}
done
