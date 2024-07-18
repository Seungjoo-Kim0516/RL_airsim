import os
import glob
import time
from datetime import datetime
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np

from agents.PPO import PPO
from environment.DroneEnv import DRLEnvironment
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

state_save_path = 'D:/KSJ/Workspace/Altitude_20240215/Flight_Data/StateData_17/'
rotor_save_path = 'D:/KSJ/Workspace/Altitude_20240215/Flight_Data/RotorData_17/'

def train():
   
    env_name = "DRL"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1501                  # max timesteps in one episode
    max_training_timesteps = 10000000   # break training loop if timeteps > max_training_timesteps

    save_model_freq = 1000          # save model frequency (in num timesteps)

    action_std = 1.0                  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.002        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 5000  # action_std decay frequency (in num timesteps)
    
    update_timestep = max_ep_len * 2       # update policy every n timesteps
    K_epochs = 60               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # Higher discount factor for continuos actions

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    random_seed = 47         # set random seed if required (0 = no random seed)
   
    env = DRLEnvironment(viz_image_cv2=False, observation_type="images")
    
    state_dim = 4108
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0]
    else:
        action_dim = env.action_space[0]
        
    run_num_pretrained = 17     #### change this to prevent overwriting weights in same env_name folder
    continue_training = False

    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("Model Directory is : " + checkpoint_path)
    
    if random_seed:        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    
    ## @@@@@@@@@@@@@@리워드 요소 별로 에피소드마다 저장할 것
    
    
    time_step = 1
    i_episode = 1
    
    best_reward = 0
    
    model_to_be_continued = 'D:/KSJ/Workspace/Altitude_20240215/models/DRL/'
    if continue_training:
        print("Bringing model :  PPO_DRL_47_16_940_best")
        # print("Bringing model : " + checkpoint_path)
        ppo_agent.load(model_to_be_continued + "PPO_DRL_47_16_940_best.pth")
    
    while time_step <= max_training_timesteps:
        
        state = env.start_race()
        current_ep_reward = 0
        
        state_data = pd.DataFrame(columns=['timestamp','pos_x', 'pos_y', 'pos_z',
                                        'vel_x', 'vel_y', 'vel_z',
                                        'ori_x', 'ori_y', 'ori_z',
                                        'ang_vel_x','ang_vel_y','ang_vel_z'])
    
        rotor_data = pd.DataFrame(columns=['timestamp', 'rotor1_speed', 'rotor1_thrust', 'rotor1_torque',
                                   'rotor2_speed', 'rotor2_thrust', 'rotor2_torque',
                                   'rotor3_speed', 'rotor3_thrust', 'rotor3_torque',
                                   'rotor4_speed', 'rotor4_thrust', 'rotor4_torque'])
        
        for t in range(1, max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state)
            action = (action+1)/2
            # action = np.float64(np.clip(action, 0.60, 0.99))


            # action = np.float64(action)*0.1
            state, reward, done = env.step(action)
            
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step +=1
            current_ep_reward += reward
            
            # Drone state 저장
            drone_state = env.airsim_client.getMultirotorState()
            pose = drone_state.kinematics_estimated.position
            vel = drone_state.kinematics_estimated.linear_velocity
            rotation = drone_state.kinematics_estimated.orientation
            ang_vel = drone_state.kinematics_estimated.angular_velocity
            
            full_state = [drone_state.timestamp, pose.x_val, pose.y_val, pose.z_val,
                          vel.x_val, vel.y_val, vel.z_val,
                          rotation.x_val, rotation.y_val, rotation.z_val,
                          ang_vel.x_val, ang_vel.y_val, ang_vel.z_val]
            state_data.loc[len(state_data)] = full_state
            
            
            # Rotor state 저장
            rotor_state = env.airsim_client.getRotorStates()
            rotors = rotor_state.rotors
            row_data = [rotor_state.timestamp]
            for rotor in rotors:
                row_data.extend([rotor['speed'], rotor['thrust'], rotor['torque_scaler']])
            rotor_data.loc[len(rotor_data)] = row_data
            
            
            # # update PPO agent
            if time_step % update_timestep == 0:
                print("Agent updated !")
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                
            # save model weights
            # if time_step % save_model_freq == 0:               
            #     ppo_agent.save(checkpoint_path)
            #     print("Model saved !")
            #     print("Total time elapsed : ", datetime.now().replace(microsecond=0) - start_time)
            #     print("--------------------------------------------------------------------------------------------")
            if t==max_ep_len:
                print("Too much steps")
            
            # break; if the episode is over
            if done or t == max_ep_len:
                if i_episode % 10 == 0:
                    # 에피소드 번호를 포함한 새로운 모델 파일 경로 생성
                    checkpoint_path_episode = directory + "PPO_{}_{}_{}_ep{}.pth".format(env_name, random_seed, run_num_pretrained, i_episode)
                    ppo_agent.save(checkpoint_path_episode)
                    # print(f"Model saved at episode {i_episode}!")
                    # print("Total time elapsed : ", datetime.now().replace(microsecond=0) - start_time)
                    # print("--------------------------------------------------------------------------------------------")
                if current_ep_reward > best_reward:
                    best_model_path = directory + "PPO_{}_{}_{}_{}_best.pth".format(env_name, random_seed, run_num_pretrained, i_episode)
                    ppo_agent.save(best_model_path)
                    print(f"New best model saved with reward: {current_ep_reward} at episode {i_episode}")
                    best_reward = current_ep_reward
                state_filename = f'episode_{i_episode}_state_data.csv'
                rotor_filename = f'episode_{i_episode}_rotor_data.csv'
                if not os.path.exists(state_save_path):
                    os.makedirs(state_save_path)
                if not os.path.exists(rotor_save_path):
                    os.makedirs(rotor_save_path)
                    
                state_data.to_csv(os.path.join(state_save_path, state_filename), index=False)
                rotor_data.to_csv(os.path.join(rotor_save_path, rotor_filename), index=False)
                
                writer.add_scalar("Reward", current_ep_reward, i_episode)
                writer.flush()
                print("Epsiode : {}, Reward : {}, Timesteps : {}".format(i_episode, current_ep_reward, time_step))
                # 리워드 데이터를 엑셀 파일로 저장
                env.save_rewards_to_excel(i_episode)
                break
            
        i_episode += 1
        
        env.reset()
        time.sleep(2)
        

def test():
   
    env_name = "DRL"
    has_continuous_action_space = True
    max_ep_len = 300           # max timesteps in one episode
    action_std = 0.10          # set same std for action distribution which was used while saving

    total_test_episodes = 5    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0001           # learning rate for actor
    lr_critic = 0.0001           # learning rate for critic

    env = DRLEnvironment(viz_image_cv2=False, observation_type="images")

    # state space dimension
    state_dim = 12

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0]
    else:
        action_dim = env.action_space[0]

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 47          #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 9      #### set this to load a particular checkpoint num

    directory = "models" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("Model is : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.start_race()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
        env.reset()               
        time.sleep(3)
    
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("Recompensa promedio : " + str(avg_test_reward))

    print("============================================================================================")
    
    
def main(args):
    if args.mode == 'train':
        train()
         
    if args.mode == 'test':
        test()

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "test",
            
        ],
        default="train",
    )
    
    args = parser.parse_args()
    main(args)
