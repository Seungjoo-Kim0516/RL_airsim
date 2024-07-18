import airsim
import numpy as np
import cv2
import threading
import time
import random
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import io

# Pretrained models were trained by movebyvelocityzAsync API
# which means action space = (2,)

class DRLEnvironment(object):
    action_space = (4,)
    
    def __init__(self, drone_name="drone_1", viz_image_cv2=True, observation_type="images"):
        self.drone_name = drone_name
        self.viz_image_cv2 = viz_image_cv2
        
        self.airsim_client = airsim.MultirotorClient()
        
        # reward
        self.max_distance = 4.5
        self.min_distance = 0.2
        self.has_collission = False
        
        # timer
        self.start_time = time.time()

        self.step_num = 0
        self.passed_num = 0
        
        self.rewards_tracking = {'z_reward': [], 'xy_reward': [], 'passing': [], 'tracking' : [], 
                                 'stability': [], 'penalty': [], 'progression_reward' : [], 'total': []}
        
        self.r = 3.0
        self.waypoints = self.eight_figure_trajectory() # 웨이포인트 생성
        self.current_waypoint_index = 0                 # 웨이포인트 인덱스 초기화
        self.flight_path=[]
        self.previous_state = None
                
    def start_race(self):
        self.airsim_client.enableApiControl(True)
        self.airsim_client.armDisarm(True)
        self.airsim_client.moveToPositionAsync(0,0,-3.0,2).join()
        time.sleep(1)
        self.start_time=time.time()
        #self.get_ground_truth_gate_poses()
        
        return self.get_state()
    
    def get_state(self):
        drone_state = self.get_drone_state()

        return drone_state
    
    def get_drone_state(self):
        drone_state = self.airsim_client.getMultirotorState()
        pose = drone_state.kinematics_estimated.position
        vel = drone_state.kinematics_estimated.linear_velocity
        ori = drone_state.kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(ori)
        ang_vel = drone_state.kinematics_estimated.angular_velocity
        Full_state = [pose.x_val, pose.y_val, pose.z_val,
                      vel.x_val, vel.y_val, vel.z_val,
                      pitch, roll, yaw,
                      ang_vel.x_val, ang_vel.y_val, ang_vel.z_val]
        return Full_state
    
    def get_previous_drone_state(self):
        state = self.previous_state
        return state
    
    def calculate_reward(self):
        # 드론 상태
        current_state = self.get_drone_state()
        prev_state = self.get_previous_drone_state()
        curr_position = np.array([current_state[0], current_state[1], current_state[2]])
        prev_position = np.array([prev_state[0], prev_state[1], prev_state[2]])
        curr_x = curr_position[0]
        curr_y = curr_position[1]
        curr_z = curr_position[2]
        body_rate = np.array([current_state[9], current_state[10], current_state[11]])
        # 웨이포인트 정보
        self.waypoints = self.eight_figure_trajectory()
        # target_waypoint = self.waypoints[self.current_waypoint_index]
        # target_position = np.array(target_waypoint[:3])
        target_waypoint = self.waypoints
        target_position = target_waypoint[:3]
        target_x = target_position[0]
        target_y = target_position[1]
        target_z = target_position[2]
        
        done = False
        
        reward = 0
        tracking = 0
        penalty = 0
        stability = 0
        passing=0
        
        z_reward = 0
        xy_reward = 0
        progression_reward=0
        
        z_weight = 8.0
        xy_weight = 2.5
        body_rate_weight = 0.5

        
        curr_distance = np.linalg.norm(curr_position - target_position)
        prev_distance = np.linalg.norm(prev_position - target_position)
        

        # 거리 보상
        z_reward -= np.sqrt(z_weight * (np.power((target_z - curr_z),2)))
        xy_reward -= np.sqrt(xy_weight * ((np.power((target_x - curr_x),2)) + (np.power((target_y - curr_y),2))))
        # 안정성 보상
        stability -= body_rate_weight*(np.sqrt(np.power(body_rate[0],2) + np.power(body_rate[1],2) + np.power(body_rate[2],2)))

        tracking += xy_reward + z_reward
          
        # 타겟 포인트와 너무 멀어졌을 때
        if curr_distance > self.max_distance:
            print("Far from the target point")
            penalty-=500
            done=True
        # 비행하고 있지 않을 때
        if curr_position[2]>0:
            print("Not Flying")
            penalty-=1000
            done=True
        
        # 마지막 step 보상
        if done==True:
            reward += (self.step_num)*2.5
        
            
        
        reward += tracking + stability + penalty + passing
        
        self.rewards_tracking['z_reward'].append(z_reward)
        self.rewards_tracking['xy_reward'].append(xy_reward)
        self.rewards_tracking['tracking'].append(tracking)
        self.rewards_tracking['stability'].append(stability)
        self.rewards_tracking['penalty'].append(penalty)
        self.rewards_tracking['passing'].append(passing)
        self.rewards_tracking['progression_reward'].append(progression_reward)
        self.rewards_tracking['total'].append(reward)
        return (reward, done)

    
    def step(self, action):
        self.previous_state = self.get_drone_state()
        action = np.float64(action)
        roll_rate = 0.25*action[0].astype(np.float64)
        pitch_rate = 0.25*action[1].astype(np.float64)
        yaw_rate = 0.1*action[2].astype(np.float64)
        throttle = ((action[3]+1)/2).astype(np.float64)
        throttle = np.float64(np.clip(throttle, 0.73, 0.99))
        
        # rotor_1 = ((action[0]+1)/2).astype(np.float64)
        # rotor_1 = np.float64(np.clip(rotor_1, 0.56, 0.99))
        # rotor_2 = ((action[1]+1)/2).astype(np.float64)
        # rotor_2 = np.float64(np.clip(rotor_2, 0.56, 0.99))
        # rotor_3 = ((action[2]+1)/2).astype(np.float64)
        # rotor_3 = np.float64(np.clip(rotor_3, 0.56, 0.99))
        # rotor_4 = ((action[3]+1)/2).astype(np.float64)
        # rotor_4 = np.float64(np.clip(rotor_4, 0.56, 0.99))
        # x = action[0].astype(np.float64)
        # y = action[1].astype(np.float64)
        # z = action[2].astype(np.float64)
        # action[3] = (np.float64(action[3])+1) / 2
        # w = np.float64(np.clip(action[3], 0.59, 0.62))

        done = False
        # self.airsim_client.moveByMotorPWMsAsync(rotor_1,
        #                                         rotor_2,
        #                                         rotor_3,
        #                                         rotor_4,
        #                                         0.01).join()
        self.airsim_client.moveByAngleRatesThrottleAsync(roll_rate,
                                                         pitch_rate,
                                                         yaw_rate,
                                                         throttle,
                                                         0.01).join()
        # self.airsim_client.moveByVelocityZAsync(x,
        #                                         y,
        #                                         -5.0,
        #                                         0.01,
        #                                         airsim.DrivetrainType.MaxDegreeOfFreedom,
        #                                         airsim.YawMode(is_rate=True, yaw_or_rate=0)).join()

                                                
        self.step_num += 1
        
        (reward, done) = self.calculate_reward()
        if done==True:
            print("Passed points : ",self.passed_num)
            print("Next waypoint was : ", (self.step_num+1))
        new_state = self.get_state()
        
        return new_state, reward, done
    
    # 리워드 요소 별 엑셀 저장
    def save_rewards_to_excel(self, episode_number):
        rewards_save_path = "D:/KSJ/Workspace/EightFigure_20240222/rewards/New/run_num_pretrained_1"
        if not os.path.exists(rewards_save_path):
            os.makedirs(rewards_save_path)
        
        filename = os.path.join(rewards_save_path, f'episode_{episode_number}_rewards.xlsx')
        df = pd.DataFrame(self.rewards_tracking)
        df.to_excel(filename, index=False)
        # print(f'Rewards saved to {filename}')

        # 리워드 데이터 리셋
        self.rewards_tracking = {'z_reward': [], 'xy_reward': [], 'passing': [], 'tracking' : [], 
                                 'stability': [], 'penalty': [], 'progression_reward' : [],'total': []}
    
    def reset(self):
        self.start_time = time.time()
        self.airsim_client.reset()
        self.current_waypoint_index = 0
        self.previous_state = None
        self.flight_path = []  # 드론의 비행 경로를 저장할 리스트 초기화
        self.rewards_tracking = {'z_reward': [], 'xy_reward': [], 'passing': [], 'tracking' : [], 
                                 'stability': [], 'penalty': [], 'progression_reward' : [], 'total': []}
        self.step_num = 0
        self.passed_num = 0

    # 8자 트랙 웨이포인트 찍기
    def eight_figure_trajectory(self):
        waypoints = []
        r = self.r
        x = r * np.sin((np.pi/5)*self.step_num*0.01)
        y = r * np.sin((np.pi/5)*self.step_num*0.01)*np.cos((np.pi/5)*self.step_num*0.01)
        z= -3.0
        waypoints = [x,y,z]
        return waypoints
    
