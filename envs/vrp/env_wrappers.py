"""
Custom environment wrappers for VRP.
Handles heterogeneous observations and share_obs properly.
"""

import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents."""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VRPVecEnv(ABC):
    """Abstract vectorized environment for VRP."""
    closed = False

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        pass


class VRPDummyVecEnv(VRPVecEnv):
    """Dummy vectorized environment that runs envs sequentially."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VRPVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space
        )

    def reset(self):
        """Reset all environments and return observations."""
        obs_list = []
        for env in self.envs:
            obs_n = env.reset()
            obs_list.append(obs_n)

        # Stack observations: shape (n_envs, n_agents, obs_dim)
        obs = np.array(obs_list)
        return obs

    def step(self, actions):
        """Step all environments."""
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs_n, reward_n, done_n, info_n = env.step(actions[i])
            obs_list.append(obs_n)
            reward_list.append(reward_n)
            done_list.append(done_n)
            info_list.append(info_n)

            # Auto-reset if all agents done
            if all(done_n):
                obs_n = env.reset()
                obs_list[-1] = obs_n

        obs = np.array(obs_list)
        rewards = np.array(reward_list)
        dones = np.array(done_list)

        return obs, rewards, dones, info_list

    def close(self):
        for env in self.envs:
            env.close()


def vrp_worker(remote, parent_remote, env_fn_wrapper):
    """Worker function for subprocess vectorized environment."""
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs_n, reward_n, done_n, info_n = env.step(data)
            if all(done_n):
                obs_n = env.reset()
            remote.send((obs_n, reward_n, done_n, info_n))
        elif cmd == 'reset':
            obs_n = env.reset()
            remote.send(obs_n)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'get_share_obs':
            # Get share_obs from the environment
            share_obs = env._get_share_obs()
            remote.send(share_obs)
        else:
            raise NotImplementedError


class VRPSubprocVecEnv(VRPVecEnv):
    """Subprocess vectorized environment for VRP."""

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=vrp_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        VRPVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.array(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
