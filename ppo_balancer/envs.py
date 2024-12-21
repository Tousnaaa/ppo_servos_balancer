#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FrameStack, RescaleAction
from settings import EnvSettings
from upkie.envs import UpkieGroundVelocity
from upkie.envs.wrappers import (
    AddActionToObservation,
    AddLagToAction,
    DifferentiateAction,
    NoisifyAction,
    NoisifyObservation,
)

class UpkieServosFlattenedWrapper(gymnasium.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = self.flatten_space(env.observation_space)
        
        self.action_space = self.flatten_space(env.action_space)
    def flatten_space(self,space):
        if isinstance(space, gymnasium.spaces.Dict):
            low = np.concatenate([np.concatenate([space[joint_name][key].low for key in space[joint_name].spaces.keys()]) for joint_name in space.spaces.keys()])
            high = np.concatenate([np.concatenate([space[joint_name][key].high for key in space[joint_name].spaces.keys()]) for joint_name in space.spaces.keys()])
            return gymnasium.spaces.Box(low=low, high =high,dtype=np.float32)
        else:
            raise ValueError("Not a dict, relis ton code")
    def flatten(self,data):
        
        
            
        return np.concatenate([np.concatenate([data[joint_name][key].ravel() for key in data[joint_name].keys()])for joint_name in data.keys()])
    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)
        return self.flatten(obs),info
    def unflatten_action(self, action):
        """
        Converts a flattened action ndarray back into the nested dictionary format.

        Args:
            action (np.ndarray): Flattened action array.

        Returns:
            dict: Unflattened action dictionary matching the nested structure of the original action_space.
        """
        action_dict = {}
        start_idx = 0
        for joint_name, joint_space in self.env.action_space.spaces.items():
            action_dict[joint_name] = {}
            for key, sub_space in joint_space.spaces.items():
                end_idx = start_idx + np.prod(sub_space.shape)
                action_dict[joint_name][key] = action[start_idx:end_idx].reshape(sub_space.shape)
                start_idx = end_idx
        
        return action_dict
    def step(self,action):
        action_dict = self.unflatten_action(action)
        
        obs, reward,done,info,last = self.env.step(action_dict)
        
        flattened_obs = self.flatten(obs)
        return flattened_obs,reward,done,info,last
    
    
    
class UpkieServosFlattenedWrapperActionShaping(gymnasium.Wrapper):  
    def __init__(self,env):
        super().__init__(env)
        self.crouch_range = (-1.26, 1.26)  # Example range for crouching (normalized)
        self.wheel_velocity_range = (-111.0, 111.0)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([self.crouch_range[0], self.wheel_velocity_range[0]]),
            high=np.array([self.crouch_range[1], self.wheel_velocity_range[1]]),
            dtype=np.float32
        )
        self.observation_space = self.flatten_space(env.observation_space)
    def flatten_space(self,space):
        if isinstance(space, gymnasium.spaces.Dict):
            low = np.concatenate([np.concatenate([space[joint_name][key].low for key in space[joint_name].spaces.keys()]) for joint_name in space.spaces.keys()])
            high = np.concatenate([np.concatenate([space[joint_name][key].high for key in space[joint_name].spaces.keys()]) for joint_name in space.spaces.keys()])
            return gymnasium.spaces.Box(low=low, high =high,dtype=np.float32)
        else:
            raise ValueError("Not a dict, relis ton code")
    def flatten(self,data):
        
        
            
        return np.concatenate([np.concatenate([data[joint_name][key].ravel() for key in data[joint_name].keys()])for joint_name in data.keys()])
    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)
        return self.flatten(obs),info
    def step(self, action):
        """
        Override the step method to implement action shaping for crouching and wheel velocity.

        Args:
            action (np.ndarray): Action array with crouching and wheel velocity values.

        Returns:
            tuple: Flattened observation, reward, done, info, and last.
        """
        crouch = action[0]
        wheel_velocity = action[1]

        # Map the crouch and wheel velocity to the full joint action dictionary
        action_dict = {}
        for joint_name in self.env.action_space.spaces.keys():
            action_dict[joint_name] = {}
            for k in self.env.action_space[joint_name].keys():
                if "hip" in joint_name and k == "position":
                    action_dict[joint_name]["position"] = crouch
                elif "knee" in joint_name and k=="position":
                    action_dict[joint_name]["position"] = 2*crouch
                elif "wheel" in joint_name and k=="velocity":
                    action_dict[joint_name]["velocity"] = wheel_velocity
                else:
                    action_dict[joint_name][k] = 0.0
            
        # Step the environment
        obs, reward, done, info, last = self.env.step(action_dict)

        # Flatten observation
        flattened_obs = self.flatten(obs)

        return flattened_obs, reward, done, info, last
    def unflatten_action(self, action):
        """
        Converts a flattened action ndarray back into the nested dictionary format.

        Args:
            action (np.ndarray): Flattened action array.

        Returns:
            dict: Unflattened action dictionary matching the nested structure of the original action_space.
        """
        action_dict = {}
        start_idx = 0
        for joint_name, joint_space in self.env.action_space.spaces.items():
            action_dict[joint_name] = {}
            for key, sub_space in joint_space.spaces.items():
                end_idx = start_idx + np.prod(sub_space.shape)
                action_dict[joint_name][key] = action[start_idx:end_idx].reshape(sub_space.shape)
                start_idx = end_idx
        print(action_dict)
        return action_dict
    def flatten(self, data):
        """
        Flattens dictionary data into a single ndarray.

        Args:
            data: The dictionary to flatten.

        Returns:
            np.ndarray: Flattened array.
        """
        return np.concatenate([
            np.concatenate([
                data[joint_name][key].ravel() for key in data[joint_name].keys()
            ])
            for joint_name in data.keys()
        ])
    
    
        
def make_training_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
) -> gymnasium.Wrapper:
    action_noise = np.array(env_settings.action_noise)
    observation_noise = np.array(env_settings.observation_noise)
    flattened_env = UpkieServosFlattenedWrapper(velocity_env)
    
    noisy_obs_env = NoisifyObservation(flattened_env, noise=observation_noise)
    noisy_env = NoisifyAction(noisy_obs_env, noise=action_noise)
    filtered_env = AddLagToAction(
        noisy_env,
        time_constant=spaces.Box(*env_settings.action_lpf),
    )
    return filtered_env


def make_accel_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
    training: bool,
) -> gymnasium.Wrapper:
    inner_env = (
        make_training_env(velocity_env, env_settings)
        if training
        else UpkieServosFlattenedWrapper(velocity_env)
    )
    history_env = FrameStack(
        AddActionToObservation(inner_env),
        env_settings.history_size,
    )
    accel_env = DifferentiateAction(
        history_env,
        min_derivative=-env_settings.max_ground_accel,
        max_derivative=+env_settings.max_ground_accel,
    )
    rescaled_accel_env = RescaleAction(
        accel_env,
        min_action=-1.0,
        max_action=+1.0,
    )
    return rescaled_accel_env

def make_ppo_balancer_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
    training: bool,
) -> gymnasium.Wrapper:
    return make_accel_env(velocity_env, env_settings, training=training)
