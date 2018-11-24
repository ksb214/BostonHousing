import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default target pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
                
        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        
        x_distance_to_target = np.linalg.norm(self.target_pos[0] - self.sim.pose[0])
        y_distance_to_target = np.linalg.norm(self.target_pos[1] - self.sim.pose[1])
        z_distance_to_target = np.linalg.norm(self.target_pos[2] - self.sim.pose[2])
        
        sum_acceleration = np.linalg.norm(self.sim.linear_accel)
        reward = (10. - distance_to_target) * 0.5 - sum_acceleration * 0.05
        
        if (x_distance_to_target > 2):
            reward += -0.05
        elif (y_distance_to_target > 2):
            reward += -0.05
        
        if (x_distance_to_target < 1):
            reward += 1
        elif (y_distance_to_target < 1):
            reward += 1
    
                 
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if done :
                reward += 5
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state