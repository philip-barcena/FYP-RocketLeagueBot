from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            # Vector from car → ball (agent-space)
            pos_diff = ball.position - car.physics.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist
            speed = np.dot(car.physics.linear_velocity, dir_to_ball)

            # Normalize, clamp to 0–1
            speed_norm = max(speed / common_values.CAR_MAX_SPEED, 0.0)

            rewards[agent] = float(speed_norm)

        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Small reward for time spent airborne (helps learn aerial touches)"""

    SCALE = 0.02  # prevents jump-spamming

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            rewards[agent] = float((not car.on_ground) * self.SCALE)

        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent when the ball moves toward the opponent goal"""

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            # Opponent goal Y coordinate (invert for orange)
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            goal_pos = np.array([0, goal_y, 0])

            # Direction from ball → goal
            pos_diff = goal_pos - ball.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_goal = pos_diff / dist

            vel = ball.linear_velocity
            vel_toward_goal = np.dot(vel, dir_to_goal)

            # Normalize 0–1
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0.0)

        return rewards
