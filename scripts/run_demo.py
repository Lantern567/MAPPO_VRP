"""
Simple demo script to run MAPPO on VRP environment and save timestep data.
Generates JSON data for visualization.
"""

import os
import sys
import json
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mappo.envs.vrp.VRP_env import VRPEnv


def run_demo(args):
    """Run a simple demo episode and save timestep data."""

    # Create environment
    env = VRPEnv(args)

    # Storage for timestep data
    episode_data = {
        'config': {
            'num_drones': args.num_drones,
            'num_customers': args.num_customers,
            'num_route_nodes': args.num_route_nodes,
            'episode_length': args.episode_length,
        },
        'route_nodes': [],
        'timesteps': []
    }

    # Store route nodes (fixed positions)
    for node in env.world.route_nodes:
        episode_data['route_nodes'].append({
            'x': float(node[0]),
            'y': float(node[1])
        })

    # Reset environment
    obs_n = env.reset()

    # Store initial customer positions (these are randomized on reset)
    initial_customers = []
    for i, c in enumerate(env.world.customers):
        initial_customers.append({
            'id': i,
            'x': float(c.state.p_pos[0]),
            'y': float(c.state.p_pos[1]),
            'time_window_start': int(c.state.time_window_start),
            'time_window_end': int(c.state.time_window_end),
            'demand': float(c.state.demand)
        })
    episode_data['customers'] = initial_customers

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < args.episode_length:
        # Record current state
        timestep_data = {
            'step': step,
            'truck': {
                'x': float(env.world.truck.state.p_pos[0]),
                'y': float(env.world.truck.state.p_pos[1]),
                'vel_x': float(env.world.truck.state.p_vel[0]),
                'vel_y': float(env.world.truck.state.p_vel[1])
            },
            'drones': [],
            'customers_served': [],
            'actions': [],
            'reward': 0.0
        }

        # Record drone states
        for i, drone in enumerate(env.world.drones):
            carrying = drone.state.carrying_package
            timestep_data['drones'].append({
                'id': i,
                'x': float(drone.state.p_pos[0]),
                'y': float(drone.state.p_pos[1]),
                'battery': float(drone.state.battery),
                'status': drone.state.status,
                'carrying_package': int(carrying) if carrying is not None else None
            })

        # Record which customers are served
        for i, c in enumerate(env.world.customers):
            if c.state.served:
                timestep_data['customers_served'].append(i)

        # Generate random actions (simple demo without trained policy)
        action_n = []
        for i, agent in enumerate(env.agents):
            # Get available actions mask
            avail = env._get_available_actions(agent)
            # Sample from available actions
            valid_actions = np.where(avail > 0)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0
            action_n.append(action)
            timestep_data['actions'].append(int(action))

        # Step environment
        obs_n, reward_n, done_n, info_n = env.step(action_n)
        done = done_n[0]
        reward = reward_n[0][0]
        total_reward += reward
        timestep_data['reward'] = float(reward)

        episode_data['timesteps'].append(timestep_data)
        step += 1

    # Add final state
    final_timestep = {
        'step': step,
        'truck': {
            'x': float(env.world.truck.state.p_pos[0]),
            'y': float(env.world.truck.state.p_pos[1]),
            'vel_x': float(env.world.truck.state.p_vel[0]),
            'vel_y': float(env.world.truck.state.p_vel[1])
        },
        'drones': [],
        'customers_served': [],
        'actions': [],
        'reward': 0.0
    }
    for i, drone in enumerate(env.world.drones):
        carrying = drone.state.carrying_package
        final_timestep['drones'].append({
            'id': i,
            'x': float(drone.state.p_pos[0]),
            'y': float(drone.state.p_pos[1]),
            'battery': float(drone.state.battery),
            'status': drone.state.status,
            'carrying_package': int(carrying) if carrying is not None else None
        })
    for i, c in enumerate(env.world.customers):
        if c.state.served:
            final_timestep['customers_served'].append(i)
    episode_data['timesteps'].append(final_timestep)

    # Summary
    episode_data['summary'] = {
        'total_steps': step,
        'total_reward': float(total_reward),
        'customers_served': sum(1 for c in env.world.customers if c.state.served),
        'total_customers': len(env.world.customers)
    }

    # Save to JSON
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'visualization', 'vrp', 'public')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'episode_data.json')

    with open(output_path, 'w') as f:
        json.dump(episode_data, f, indent=2)

    print(f"Episode completed!")
    print(f"  Steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Customers served: {episode_data['summary']['customers_served']}/{episode_data['summary']['total_customers']}")
    print(f"  Data saved to: {output_path}")

    return episode_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', type=str, default='truck_drone_basic')
    parser.add_argument('--num_drones', type=int, default=3)
    parser.add_argument('--num_customers', type=int, default=10)
    parser.add_argument('--num_route_nodes', type=int, default=5)
    parser.add_argument('--episode_length', type=int, default=10000)
    parser.add_argument('--delivery_threshold', type=float, default=0.05)
    parser.add_argument('--recovery_threshold', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)

    run_demo(args)


if __name__ == '__main__':
    main()