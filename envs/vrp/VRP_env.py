"""
VRP Environment Factory.
Creates MultiAgentVRPEnv instances with appropriate scenario.
"""

from mappo.envs.vrp.environment import MultiAgentVRPEnv
from mappo.envs.vrp.scenarios import load


def VRPEnv(args):
    """
    Create a MultiAgentVRPEnv with the specified scenario.

    Args:
        args: Namespace containing:
            - scenario_name: Name of scenario file (without .py extension)
            - num_drones: Number of drone agents
            - num_customers: Number of customers
            - num_route_nodes: Number of route nodes for truck
            - episode_length: Max episode steps
            - delivery_threshold: Distance threshold for delivery
            - recovery_threshold: Distance threshold for drone recovery

    Returns:
        MultiAgentVRPEnv instance
    """
    # Load scenario
    scenario_name = getattr(args, 'scenario_name', 'truck_drone_basic')
    scenario = load(scenario_name + ".py").Scenario()

    # Create world
    world = scenario.make_world(args)

    # Create environment with callbacks
    env = MultiAgentVRPEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.compute_global_reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info,
        done_callback=scenario.is_terminal,
        available_actions_callback=scenario.get_available_actions,
        share_obs_callback=scenario.get_share_obs,
        discrete_action=True
    )

    return env
