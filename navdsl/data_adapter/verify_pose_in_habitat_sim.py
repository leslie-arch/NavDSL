#!/usr/bin/env python3
"""Highest-confidence pose verification: load a real HM3D scene in habitat-sim,
teleport the agent to a converted viewpoint position, render RGB, save images.

If the images show plausible room geometry (not inside a wall, not upside down),
the pose coordinate mapping (X=pose[3], Y=pose[11], Z=-pose[7]) is correct.

Run on the remote host that has habitat-sim + HM3D meshes:
  python -m navdsl.data_adapter.verify_pose_in_habitat_sim \
      --episodes data/datasets/vln/hm3d/autovln/v1/train/train.json.gz \
      --scene-dataset /sata/sdb7/.../hm3d_annotated_basis.scene_dataset_config.json \
      --scenes-dir /sata/sdb7/.../versioned_data/hm3d-0.2/hm3d/ \
      --output-dir /tmp/pose_check
"""
import argparse
import gzip
import json
import os
import sys

# Import lazily so the script can still print --help without habitat installed.
def _import_habitat():
    try:
        import habitat_sim  # type: ignore
        import magnum  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        return habitat_sim, magnum, np, Image
    except ImportError as e:
        print(f"ERROR: this script needs habitat_sim, magnum, numpy, PIL. {e}", file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--scene-dataset", required=True, help="Path to .scene_dataset_config.json")
    ap.add_argument("--scenes-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-episodes", type=int, default=1)
    ap.add_argument("--views-per-episode", type=int, default=5)
    args = ap.parse_args()

    habitat_sim, magnum, np, Image = _import_habitat()

    os.makedirs(args.output_dir, exist_ok=True)

    with gzip.open(args.episodes, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"][: args.num_episodes]

    # Build one simulator config per scene (each episode may be in a different scene)
    for ep in episodes:
        scan = ep["scene_scan_id"]
        scene_id = ep["scene_id"]  # e.g. hm3d/train/{scan}/{scan}.basis.glb
        full_scene_path = os.path.join(args.scenes_dir, scene_id)
        print(f"Episode {ep['episode_id']} scan={scan}", file=sys.stderr)
        print(f"  scene path: {full_scene_path}", file=sys.stderr)
        print(f"  scene path partial: {args.scenes_dir} + {scene_id}", file=sys.stderr)
        if not os.path.isfile(full_scene_path):
            print(f"  ERROR: scene file missing", file=sys.stderr)
            continue

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = full_scene_path
        sim_cfg.scene_dataset_config_file = args.scene_dataset
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.5
        sensor = habitat_sim.sensor.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [480, 640]
        sensor.position = [0.0, 1.5, 0.0]
        sensor.hfov = 90
        agent_cfg.sensor_specifications = [sensor]
        # Mandatory action space
        agent_cfg.action_space = {}
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        sim = habitat_sim.Simulator(cfg)

        # Walk through the first N viewpoints of the reference path
        ref_vps = ep["reference_viewpoints"][: args.views_per_episode]
        for i, (vp_id, pos, rot) in enumerate(zip(
            ref_vps,
            ep["reference_path"][: args.views_per_episode],
            [ep["start_rotation"]] + [ep["start_rotation"]] * (len(ref_vps) - 1),  # rotation optional
        )):

            agent_keys = sim.config.agents
            print(f'Simulator agents: {agent_keys}')
            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            print(f'Simulator agents: \n{agent_state}')
            agent_state.position = np.array(pos, dtype=np.float32)
            # Quaternion (qx,qy,qz,qw) -> magnum.Quaternion
            qx, qy, qz, qw = rot
            print(f"rotation: {qx} {qy} {qz} {qw}")
            agent_state.rotation = (qx, qy, qz, qw)
            # agent_state.rotation = magnum.Quaternion(
            #     magnum.Vector3(qx, qy, qz), qw
            # )
            agent.set_state(agent_state)

            obs = sim.get_sensor_observations()
            rgb = obs["rgb"]
            out_path = os.path.join(
                args.output_dir,
                f"{ep['episode_id']}_vp{vp_id}_step{i}.png",
            )
            Image.fromarray(rgb).save(out_path)
            print(f"  saved {out_path}  pos={pos}", file=sys.stderr)

        sim.close()


if __name__ == "__main__":
    main()
