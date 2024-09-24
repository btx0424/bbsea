import os
os.environ['MUJOCO_GL'] = 'egl'
import time
import hydra
import logging
import termcolor
import numpy as np
import torch
from utils import make_output_dirs_and_filter_task_desc_list, make_trajectory_dir_for_each_episode, save_str_to_file, set_up_env, set_up_scene, if_early_end
from scalingup.utils.core import (
    PointCloud,
    Trajectory,
    ControlTrajectory,
    Task_BEBS,
    Action,
)
from scalingup.utils.generic import setup_logger
from perception.scene_graph import get_scene_graph
from task_propose.task_propose import propose_task
from task_decompose.task_decompose import decompose_task
from success_infer.success_infer import infer_if_success
from torchvision.io import write_video
import imageio


@hydra.main(config_path="config", config_name="bebs_pipeline", version_base="1.2")
def main(config):
    # setup_logger()
    env = set_up_scene(scene_id=config.scene_id)
    env.reset(episode_id=0)

    # get scene graph
    scene_graph = get_scene_graph(env)
    print(scene_graph)
    # propose task
    task_desc_list = propose_task(scene_graph)
    print(task_desc_list)
    task_desc_list = make_output_dirs_and_filter_task_desc_list(config, task_desc_list)
    save_str_to_file(f'scene_graph:\n{scene_graph}\n\proposed_tasks:\n{task_desc_list}', f'{config.log_path}/{config.scene_id}/proposed_tasks_{int(time.time())}.txt')

    for task_desc in task_desc_list:
        print(termcolor.colored(f"Processing task: {task_desc}", "green"))
        task = Task_BEBS(desc=task_desc)
        success_num = 0
        i_this_time = 0
        is_except = False

        for i, episode_id in enumerate(range(config.max_trajectory_number_per_task)):
            episode_id = i + 1000
            try:
                have_collected, is_the_trajectory_sucess = make_trajectory_dir_for_each_episode(f'{config.trajectory_path}/{config.scene_id}/{task_desc}/{episode_id}')
                if have_collected:
                    if is_the_trajectory_sucess:
                        success_num += 1
                    print(success_num)
                    continue
                print(success_num)
                env.reset(episode_id=episode_id)
                env.start_record()
                episode = []
    
                scene_graph_list = []
                scene_graph = get_scene_graph(env)
                scene_graph_list.append(scene_graph)
                # decompose task
                if i_this_time == 0 or 'PlaceAt' in ''.join(primitive_action_str_list) or 'Push' in ''.join(primitive_action_str_list) or is_except == True:  # parameters of PlaceAt and Push contain position which is different in different scene initialization, so the primitive_action_list should be regenerated according to the current scene configuration.
                    subtask_list, primitive_action_str_list, primitive_action_list, task_decomposition_reasoning = decompose_task(task_desc, scene_graph)
                i_this_time += 1
                # execute primitive actions
                for subtask_actions in primitive_action_list:
                    for primitive_action in subtask_actions:
                        print(f"Execute primitive {primitive_action.__class__.__name__}.")
                        sub_episode = primitive_action.do(env)
                        episode.extend(sub_episode)
                        scene_graph = get_scene_graph(env)
                        scene_graph_list.append(scene_graph)
                
                # success detect
                is_successful, infer_info = infer_if_success(task_desc, scene_graph_list)
            
                ## collect trajectories (and videos for visualization)
                trajectory = Trajectory(episode=tuple(episode), episode_id=episode_id, task=task, policy_id='bebs')
                ## trajectory.dump_video(output_path=f'{config.log_path}/{config.scene_id}/{task_desc}/{episode_id}_{is_successful}.mp4')
                control_trajectory = ControlTrajectory.from_trajectory(trajectory=trajectory)
                
                images = [] 
                states = []
                actions = []
                for control_step in control_trajectory.control_sequence:
                    rgb = control_step.obs.images["front"].rgb
                    images.append(rgb)
                    state = np.array(control_step.obs.state.end_effector_pose.flattened)
                    states.append(state)
                    control = control_step.obs.control
                    action = np.zeros(9)
                    action[0:3] = control["target_ee_action/pos"]
                    action[3:7] = control["target_ee_action/quat"]
                    action[7] = control["target_ee_action/allow_contact"]
                    action[8] = control["target_ee_action/gripper_command"]
                    actions.append(action)
                ep_len = len(control_trajectory.control_sequence)
                is_first = np.zeros(ep_len)
                is_terminal = np.zeros(ep_len)
                is_first[0] = True
                is_terminal[-1] = True

                images = torch.as_tensor(np.stack(images), dtype=torch.uint8)
                actions = torch.as_tensor(np.stack(actions), dtype=torch.float32)
                states = torch.as_tensor(np.stack(states), dtype=torch.float32)

                data_dir = f'{config.trajectory_path}/{config.scene_id}/{task_desc}/'

                episode_data = {
                    "image": images,
                    "state": states,
                    "action": actions,
                    "is_first": torch.as_tensor(is_first),
                    "is_terminal": torch.as_tensor(is_terminal)
                }
                data_path = os.path.join(data_dir, f"{episode_id}_{is_successful}.pt")
                torch.save(episode_data, data_path)

                os.makedirs(data_dir, exist_ok=True)
                video_path = os.path.join(data_dir, f'{episode_id}_{is_successful}.mp4')
                print(f"save video of len {len(images)} to {video_path}")
                imageio.mimsave(video_path, images)
                
                if is_successful: 
                    success_num += 1
                    control_trajectory.dump(path=f'{config.trajectory_path}/{config.scene_id}/{task_desc}/{episode_id}/{task_desc}_{episode_id}_{is_successful}.mdb')  # make sure there is no file with the same path
                else:
                    save_str_to_file('false', f'{config.trajectory_path}/{config.scene_id}/{task_desc}/{episode_id}/{task_desc}_{episode_id}_{is_successful}.txt')
                is_except = False
            except Exception as e:
                print(termcolor.colored('bebs_pipeline except', "red"), e)
                is_except = True
                continue
            env.end_record(f'{config.log_path}/{config.scene_id}/{task_desc}/{episode_id}_{is_successful}.mp4')
            scene_graph_list_str = ''
            primitive_action_list_str = ''
            for scene_graph in scene_graph_list:
                scene_graph_list_str += f'  ----------\n{scene_graph}'
            for primitive_action_str in primitive_action_str_list:
                primitive_action_list_str += f'{primitive_action_str}\n'
            save_str_to_file(f'task_decomposition_reasoning:\n{task_decomposition_reasoning}\n\nsubtask_list:\n{subtask_list}\n\nprimitive_action_list:\n{primitive_action_list_str}\n\nscene_graph_list:\n{scene_graph_list_str}\n\ninfer_info:\n{infer_info}', f'{config.log_path}/{config.scene_id}/{task_desc}/log_{episode_id}.txt')

            if if_early_end(success_num, i, config):
                break


if __name__ == '__main__':
    main()