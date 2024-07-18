import airsim
import numpy as np
import cv2
import threading
import time
import random
import math
import os
import pandas as pd

class DRLEnvironment(object):
    action_space = (4,)
    
    def __init__(self, drone_name="drone_1", viz_image_cv2=True, observation_type="images"):
        self.drone_name = drone_name
        self.viz_image_cv2 = viz_image_cv2
        
        self.airsim_client = airsim.MultirotorClient()
        
        # reward
        self.max_distance = 13
        self.previous_distance = 2       
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False
        
        # timer
        self.start_time = time.time()
        self.last_gate_passed_time = time.time() # 게이트 통과 시점 저장하기 위한 변수
        
        self.step_num = 0
        self.gate_pass=0
        self.gate_not_pass=0
        self.collision_count=0
        
        self.camera_name = "0"
        self.image_type = airsim.ImageType.Scene
        
        self.prev_action=None
        self.previous_state = None
        
        self.gate_poses_ground_truth = self.get_ground_truth_gate_poses()
        self.next_gate_idx = 0  # 다음 통과해야 할 게이트 인덱스
        self.gate_pass_threshold = 1.5
        
        self.gate_passed_status = [False] * len(self.gate_poses_ground_truth)  # 각 게이트의 통과 여부 추적
        self.rewards_tracking = {'progression_reward': [], 'optical_axis_reward': [], 'tracking' : [], 'stability': [], 'penalty': [],
                                 'action_difference' : [], 'body_rate' : [], 'Total' : []}
                
    def start_race(self):
        self.airsim_client.enableApiControl(True)
        self.airsim_client.armDisarm(True)
        self.airsim_client.moveToPositionAsync(0,0,-3.0,2).join()
        self.start_time=time.time()        
        return self.get_state()
    
    def get_state(self):
        drone_state = self.get_drone_state()
        
        camera_image_state = self.get_camera_image()
        state = np.concatenate((drone_state, camera_image_state.flatten()))
        return state
    
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
    
    def get_camera_image(self):
        responses = self.airsim_client.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)])
        response = responses[0]

        # 이미지 데이터가 있는지
        if response.image_data_float:
            img1d = np.array(response.image_data_float, dtype=np.float32) 
            img1d = np.clip(img1d, 0, 100)  # 깊이 값을 0에서 100미터 사이로 제한
            img1d = (img1d / 100.0) * 255.0  # 0에서 100 사이의 값을 0에서 255로 정규화
            img2d = np.reshape(img1d, (response.height, response.width))  # 2D 이미지로 재구성

            img_resized = cv2.resize(img2d, (64, 64), interpolation=cv2.INTER_AREA)  # 크기 조정
            img_normalized = img_resized.astype(np.float32) / 255.0  # 정규화
            img_normalized = np.expand_dims(img_normalized, axis=0)  # 차원 확장

            return img_normalized
        else:
            # 이미지 데이터가 없는 경우
            print("Failed to receive camera image.")
            # 빈 이미지
            return np.zeros((1, 64, 64), dtype=np.float32)
    
    def get_previous_drone_state(self):
        state = self.previous_state
        return state
    
    def calculate_reward(self, current_action, prev_action):
        current_state = self.get_drone_state()
        prev_state = self.get_previous_drone_state()
        curr_position = np.array([current_state[0], current_state[1], current_state[2]])
        prev_position = np.array([prev_state[0], prev_state[1], prev_state[2]])
        body_rate = np.array([current_state[9], current_state[10], current_state[11]])
        next_gate = self.gate_poses_ground_truth[self.next_gate_idx]
        isCollided = self.airsim_client.simGetCollisionInfo().has_collided
        
        done = False
        
        # reward 요소
        reward = 0
        tracking = 0
        penalty = 0
        stability = 0
        
        progression_reward = 0
        optical_axis_reward = 0
        gate_detect_reward = 0
        cam_reward = 0
        body_rate_reward = 0
        action_smoothing_reward = 0
        
        # Hyperparameter
        progression_weight = 10.0
        optical_axis_weight = 0.002
        angle_weight = 0.001
        body_rate_weight = 0.01
        yaw_weight = 2.0
        action_difference_weight = 0.003
        
        # 지나가야 하는 게이트 통과 여부 판단
        # 바로 다음 게이트 통과 여부, 모든 게이트 통과 여부
        isGatePass, AllGatePass = self.check_gate_passed()
        

        # 지나가야 하는 게이트 인식 여부 판단
        isGateDetected = self.check_next_gate_recognition()

        # 카메라와 게이트 중심간의 각도
        angle = self.calculate_camera_gate_angle()
        angle = np.clip(angle,0.1,179.9)
        # 액션 변화량
        action_difference = np.linalg.norm(current_action - prev_action)
        
        # 지나가야 하는 게이트 통과
        if isGatePass:
            self.start_time = time.time()
            self.gate_pass+=1
            reward+=2.5*self.gate_pass
            if AllGatePass:
                reward+=10
                done=True
        
        gate_position = np.array([next_gate.position.x_val, next_gate.position.y_val, next_gate.position.z_val])
        prev_distance = np.linalg.norm(prev_position - gate_position)
        curr_distance = np.linalg.norm(curr_position - gate_position)
        # 이전 스텝에서의 게이트 중심점까지 거리 - 현재 스텝에서 게이트 중심점까지 거리
        progression_reward += progression_weight*(prev_distance-curr_distance)
        
        # 광축 벡터가 게이트 중심점과 30도 이내 유지하게끔
        if angle <= 30:
            optical_axis_reward += optical_axis_weight * (30 - angle)
        else:
            optical_axis_reward += -angle_weight * (angle - 30)

        tracking += progression_reward + optical_axis_reward
        
        # # 지나가야 하는 게이트 인식을 잘 하는지
        # if isGateDetected:
        #     gate_detect_reward+=0.2
        
        #body rate
        body_rate_reward += -body_rate_weight*(np.sqrt(np.power(body_rate[0],2) + np.power(body_rate[1],2)) + yaw_weight*np.sqrt(np.power(body_rate[2],2)))
        # 액션 변화량 리워드
        action_smoothing_reward += -action_difference_weight * action_difference  # 액션 변화가 작을수록 리워드 감소가 적어집니다.
        
        # 비행 안정성 : body rate + action 변화량
        stability += body_rate_reward + action_smoothing_reward

        
        
        # 게이트와 너무 멀어졌을 때
        if curr_distance > self.max_distance:
            penalty-=10
            print("Far from the gate")
            done=True
            
        time_elpased = time.time() - self.start_time
        # 오랜 시간 동안 게이트 통과 실패 시
        if not isGatePass:
            if time_elpased >= 25:
                penalty-=10
                print("Too much time")
                done=True
            
        
        if time_elpased < 1:
            isCollided=False
        # 충돌
        if isCollided:
            self.collision_count+=1
        if self.collision_count>=40:
            penalty-=10
            print("Collsion")
            done=True
        # 비행 안할 때
        if curr_position[2]>0:
            penalty-=10
            print("Not Flying")
            done=True
            
        # 총 리워드
        reward += tracking + stability + penalty
        
        self.rewards_tracking['progression_reward'].append(progression_reward)
        self.rewards_tracking['optical_axis_reward'].append(optical_axis_reward)
        self.rewards_tracking['tracking'].append(tracking)
        self.rewards_tracking['stability'].append(stability)
        self.rewards_tracking['penalty'].append(penalty)
        self.rewards_tracking['action_difference'].append(action_smoothing_reward)
        self.rewards_tracking['body_rate'].append(body_rate_reward)
        self.rewards_tracking['Total'].append(reward)
        return (reward, done)
    
    def get_previous_action(self):
        return self.prev_action
    
    def step(self, action):
        # prev_action = self.get_previous_action()
        current_action = np.copy(action)
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action)
            
        self.previous_state = self.get_drone_state()
        # 구간 별 thrust 값 clip 범위 다르게 해도 됨
        action = np.float64(np.clip(action, 0.56, 0.99))
        # if self.next_gate_idx==0:
        #     action = np.float64(np.clip(action, 0.56, 0.99))
        # elif self.next_gate_idx==1:
        #     action = np.float64(np.clip(action, 0.60, 0.99))
        # elif self.next_gate_idx==2:
        #     action = np.float64(np.clip(action, 0.63, 0.99))
        # else:
        #     action = np.float64(np.clip(action, 0.56, 0.99))
        self.airsim_client.moveByMotorPWMsAsync(action[0],action[1],action[2],action[3],0.01).join()
        
        self.step_num+=1
        done = False
        (reward, done) = self.calculate_reward(current_action, self.prev_action)
        
        new_state = self.get_state()
        self.prev_action = current_action
        return new_state, reward, done
    
    def save_rewards_to_excel(self, episode_number):
        rewards_save_path = "D:/KSJ/Workspace/Altitude_20240215/rewards/run_num_pretrained_17"
        if not os.path.exists(rewards_save_path):
            os.makedirs(rewards_save_path)
        
        filename = os.path.join(rewards_save_path, f'episode_{episode_number}_rewards.xlsx')
        df = pd.DataFrame(self.rewards_tracking)
        df.to_excel(filename, index=False)
        # print(f'Rewards saved to {filename}')

        # 리워드 데이터 리셋
        self.rewards_tracking = {'progression_reward': [], 'optical_axis_reward': [], 'tracking': [], 'stability': [], 'penalty': [],
                                 'action_difference' : [], 'body_rate' : [], 'Total' : []}
    
    def reset(self):
        self.start_time = time.time()
        self.last_gate_passed_time = time.time()
        self.airsim_client.reset()
        self.max_distance = 13
        self.previous_distance = 2
        self.next_gate_idx = 0
        self.collision_count=0
        self.gate_passed_status = [False] * len(self.gate_poses_ground_truth)  # 각 게이트의 통과 여부 추적
        self.rewards_tracking = {'progression_reward': [], 'optical_axis_reward': [],'tracking' : [], 'stability': [], 'penalty': [],
                                 'action_difference' : [], 'body_rate' : [], 'Total' : []}
        self.gate_pass = 0
        self.step_num = 0
        self.prev_action=None
        self.previous_state = None

# 게이트 정보 순차적으로 나열, 게이트 통과 여부 판단

    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)        
            while (math.isnan(curr_pose.position.x_val) or math.isnan(curr_pose.position.y_val) or math.isnan(curr_pose.position.z_val)):
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(curr_pose.position.x_val), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val}"
            assert not math.isnan(curr_pose.position.y_val), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val}"
            assert not math.isnan(curr_pose.position.z_val), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val}"
            gate_poses_ground_truth.append(curr_pose)
            #print(gate_name, curr_pose.position)
        
        return gate_poses_ground_truth
    
    def quaternion_to_rotation_matrix(self,q):
        """
        쿼터니언을 회전 행렬로 변환합니다.
        """
        w, x, y, z = q
        R = np.array([[1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
                      [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
                      [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]])
        return R
    
    
    # 통과한 게이트 인덱스 반환 -> 지나가야 하는 것과 다를 경우 페널티 부가
    def get_passed_gate_index(self):
        drone_position = np.array([
            self.airsim_client.getMultirotorState().kinematics_estimated.position.x_val,
            self.airsim_client.getMultirotorState().kinematics_estimated.position.y_val,
            self.airsim_client.getMultirotorState().kinematics_estimated.position.z_val,
        ])

        closest_gate_index = None
        closest_distance = float('inf')

        # 모든 게이트를 순회하며 드론과의 거리 계산
        for idx, gate_pose in enumerate(self.gate_poses_ground_truth):
            gate_position = np.array([gate_pose.position.x_val, gate_pose.position.y_val, gate_pose.position.z_val])
            distance = np.linalg.norm(drone_position - gate_position)

            # 드론과 게이트 사이의 거리가 임계값 이내이고, 가장 가까운 거리인 경우
            if distance < self.gate_pass_threshold and distance < closest_distance:
                closest_gate_index = idx
                closest_distance = distance

        # 가장 가까이 지나간 게이트의 인덱스 반환
        return closest_gate_index
    
    def check_gate_passed(self):
        """
        현재 드론 위치에 기반하여 다음 통과해야 할 게이트가 통과되었는지 확인
        """
        if self.next_gate_idx >= len(self.gate_poses_ground_truth):
            # 첫 번째 False는 게이트를 통과하지 않았음을, 두 번째 False는 모든 게이트를 이미 통과했음을 의미
            return False, False
        
        
        drone_state = self.airsim_client.getMultirotorState()
        drone_position = np.array([
            drone_state.kinematics_estimated.position.x_val,
            drone_state.kinematics_estimated.position.y_val,
            drone_state.kinematics_estimated.position.z_val,
        ])

        # 다음 통과해야 할 게이트의 위치
        next_gate_pose = self.gate_poses_ground_truth[self.next_gate_idx].position
        next_gate_position = np.array([next_gate_pose.x_val, next_gate_pose.y_val, next_gate_pose.z_val])

        # 드론과 다음 게이트 사이의 거리 계산
        distance_to_next_gate = np.linalg.norm(drone_position - next_gate_position)

        # 게이트 통과 조건 검사 (예: 거리가 특정 임계값 이내인 경우)
        if distance_to_next_gate < self.gate_pass_threshold:
            print(f"Gate {self.next_gate_idx + 1} passed.")
            self.next_gate_idx += 1  # 다음 게이트로 인덱스 업데이트
            print(f"Next Gate is : {self.next_gate_idx + 1}")
            if self.next_gate_idx >= len(self.gate_poses_ground_truth):
                print("All Gate passed!")
                return True, True  # 첫 번째 True는 게이트를 통과했음을, 두 번째 True는 모든 게이트를 통과했음을 의미
            return True, False  # 게이트를 통과했으나 아직 모든 게이트를 통과하지 않았음
        return False, False  # 게이트를 통과하지 않았고, 모든 게이트를 통과하지도 않음
    
    
# 카메라가 바라보는 방향을 나타내는 벡터 & 현재 드론의 위치와 게이트 중심을 이은 벡터 간의 각도 구함
    
    def calculate_angle_between_vectors(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (norm_vec1 * norm_vec2)
        angle = np.arccos(np.clip(cos_theta, -0.99, 0.99))
        angle_deg = np.degrees(angle)
        return angle_deg
    
    def calculate_camera_gate_angle(self):
        # 모든 게이트 통과 시 각도 계산 안 함.
        if self.next_gate_idx >= len(self.gate_poses_ground_truth):
            return np.inf
        # 카메라의 현재 상태 및 방향 가져오기
        drone_state = self.airsim_client.getMultirotorState()
        camera_state = self.airsim_client.simGetCameraInfo(self.camera_name)
        camera_orientation = camera_state.pose.orientation
        # 카메라 방향
        camera_rotation_matrix = self.quaternion_to_rotation_matrix((camera_orientation.w_val,
                                                                     camera_orientation.x_val,
                                                                     camera_orientation.y_val,
                                                                     camera_orientation.z_val))
        camera_direction = np.dot(camera_rotation_matrix, np.array([1, 0, 0]))  # 카메라가 x축 방향을 향함

        # 다음 게이트 중심점 방향
        next_gate_pose = self.gate_poses_ground_truth[self.next_gate_idx].position
        next_gate_position = np.array([next_gate_pose.x_val, next_gate_pose.y_val, next_gate_pose.z_val])
        drone_position = np.array([drone_state.kinematics_estimated.position.x_val,
                                   drone_state.kinematics_estimated.position.y_val,
                                   drone_state.kinematics_estimated.position.z_val])
        gate_direction = next_gate_position - drone_position

        # 두 벡터 사이의 각도 계산
        angle_between_camera_and_gate = self.calculate_angle_between_vectors(camera_direction, gate_direction)
        return angle_between_camera_and_gate
    
# 게이트 인식하는지 판단 --> reward에 추가해봄직한 요소
    def check_next_gate_recognition(self):
        # 카메라에서 이미지 캡처
        rawImage = self.airsim_client.simGetImage(self.camera_name, self.image_type)
        if not rawImage:
            return False  # 이미지가 없으면 False 반환

        # 이미지를 OpenCV 포맷으로 변환
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)

        # 카메라 시야 내의 게이트 바운딩 박스 정보 가져오기
        Gates = self.airsim_client.simGetDetections(self.camera_name, self.image_type)
        if Gates:
            for gate in Gates:
                # 현재 지나가야 하는 게이트의 이름 확인
                if gate.name == f"Gate{self.next_gate_idx}":
                    # 바운딩 박스 그리기 (옵션)
                    cv2.rectangle(png, (int(gate.box2D.min.x_val), int(gate.box2D.min.y_val)),
                                  (int(gate.box2D.max.x_val), int(gate.box2D.max.y_val)), (255, 0, 0), 2)
                    cv2.putText(png, gate.name, (int(gate.box2D.min.x_val), int(gate.box2D.min.y_val - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12))

                    # 지나가야 하는 다음 게이트가 인식된 경우
                    return True

        # 지나가야 하는 다음 게이트가 인식되지 않은 경우
        return False
