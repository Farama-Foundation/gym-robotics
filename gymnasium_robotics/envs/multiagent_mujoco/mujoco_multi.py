"""Main file for MaMuJoCo includes the MultiAgentMujocoEnv class.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/mujoco_multi.py)
Original Author: Schroeder de Witt

Then Modified by @Kallinteris-Andreas for this project
changes:
 - General code cleanup, factorization, type hinting, adding documentation and code comments.
 - Now uses PettingZoo APIs instead of an original API.
 - Now supports custom agent factorizations.
 - Added `gym_env` argument, which can be used to load third party `Gymnasium.MujocoEnv` environments.

This project is covered by the Apache 2.0 License.
"""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import pettingzoo
from gymnasium.wrappers import TimeLimit

import gymnasium_robotics.envs.multiagent_mujoco.many_segment_ant as many_segment_ant
import gymnasium_robotics.envs.multiagent_mujoco.many_segment_swimmer as many_segment_swimmer
from gymnasium_robotics.envs.multiagent_mujoco.coupled_half_cheetah import (
    CoupledHalfCheetahEnv,
)
from gymnasium_robotics.envs.multiagent_mujoco.obsk import (
    build_obs,
    get_joints_at_kdist,
    get_parts_and_edges,
)

# TODO for future revisions v2?
# color the renderer
# support other Gymnasium-Robotics MuJoCo environments

_MUJOCO_GYM_ENVIROMENTS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "HumanoidStandup",
    "Humanoid",
    "Reacher",
    "Swimmer",
    "Pusher",
    "Walker2d",
    "InvertedPendulum",
    "InvertedDoublePendulum",
]


class MultiAgentMujocoEnv(pettingzoo.utils.env.ParallelEnv):
    """Class for multi agent factorizing mujoco environments.

    Doc can be found at (https://robotics.farama.org/envs/mamujoco/)
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "name": "MaMuJoCo",
        "is_parallelizable": True,
        # "render_fps": 0,  #depends on underlying Envrioment
        "has_manual_policy": False,
    }

    def __init__(
        self,
        scenario: str,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        agent_factorization: dict[str, any] | None = None,
        local_categories: list[list[str]] | None = None,
        global_categories: tuple[str, ...] | None = None,
        render_mode: str | None = None,
        gym_env: gymnasium.envs.mujoco.mujoco_env.MujocoEnv | None = None,
        **kwargs,
    ):
        """Init.

        Args:
            scenario: The Task/Environment, valid values:
                "Ant", "HalfCheetah", "Hopper", "HumanoidStandup", "Humanoid", "Reacher", "Swimmer", "Pusher", "Walker2d", "InvertedPendulum", "InvertedDoublePendulum", "ManySegmentSwimmer", "ManySegmentAnt", "CoupledHalfCheetah"
            agent_conf: Typical values:
                '${Number Of Agents}x${Number Of Segments per Agent}${Optionally Additional options}', eg '1x6', '2x4', '2x4d',
                If it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
            agent_obsk: Number of nearest joints to observe,
                If set to 0 it only observes local state,
                If set to 1 it observes local state + 1 joint over,
                If set to 2 it observes local state + 2 joints over,
                If it set to None the task becomes single agent (the agents observe the entire environment)
                The Default value is: 1
            agent_factorization: A custom factorization of the MuJoCo environment (overwrites agent_conf),
                see DOC [how to create new agent factorizations](https://robotics.farama.org/envs/MaMuJoCo/index.html#how-to-create-new-agent-factorizations).
            local_categories: The categories of local observations for each observation depth,
                It takes the form of a list where the k-th element is the list of observable items observable at the k-th depth
                For example: if it is set to `[["qpos, qvel"], ["qvel"]]` then means each agent observes its own position and velocity elements, and it's neighbors velocity elements.
                The default is: Check each environment's page on the "observation space" section.
            global_categories: The categories of observations extracted from the global observable space,
                For example: if it is set to `("qpos")` out of the globally observable items of the environment, only the position items will be observed.
                The default is: `("qpos", "qvel")`
            render_mode: See [Gymnasium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/),
                valid values: 'human', 'rgb_array', 'depth_array'
            gym_env: A custom `MujocoEnv` environment, overrides generation of environment by `MaMuJoCo`.
            kwargs: Additional arguments passed to the [Gymnasium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) environment,
                Note: arguments that change the observation space will not work.

            Raises: NotImplementedError: When the scenario is not supported (not part of of the valid values).
        """
        # Create underlying single agent environment
        if gym_env is None:
            self.single_agent_env = self._create_base_gym_env(
                scenario, agent_conf, render_mode, **kwargs
            )
        else:
            self.single_agent_env = gym_env

        if agent_conf is None:
            self.agent_obsk = None
        else:
            self.agent_obsk = agent_obsk  # if None, fully observable else k>=0 implies observe nearest k agents or joints

        # load the agent factorization
        if agent_factorization is None:
            (
                self.agent_action_partitions,
                mujoco_edges,
                self.mujoco_globals,
            ) = get_parts_and_edges(scenario, agent_conf)
        else:
            self.agent_action_partitions = agent_factorization["partition"]
            mujoco_edges = agent_factorization["edges"]
            self.mujoco_globals = agent_factorization["globals"]

        # Create agent lists
        self.possible_agents = [
            "agent_" + str(agent_id)
            for agent_id in range(len(self.agent_action_partitions))
        ]
        self.agents = self.possible_agents

        # load the observation categories (from init arguments or generate them)
        if local_categories is None:
            self.local_categories = self._generate_local_categories(scenario)
        else:
            self.local_categories = local_categories
        if global_categories is None:
            self.global_categories = ("qpos", "qvel")
        else:
            self.global_categories = global_categories

        # load the observations per depth level
        if self.agent_obsk is not None:
            self.k_dicts = [
                get_joints_at_kdist(
                    self.agent_action_partitions[agent_id],
                    mujoco_edges,
                    k=self.agent_obsk,
                )
                for agent_id in range(self.num_agents)
            ]

        self.observation_factorization = self.create_observation_mapping()

        # Create observation and action spaces
        self.observation_spaces, self.action_spaces = {}, {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            self.action_spaces[self.possible_agents[agent_id]] = gymnasium.spaces.Box(
                low=self.single_agent_env.action_space.low[0],
                high=self.single_agent_env.action_space.high[0],
                shape=(len(partition),),
                dtype=np.float32,
            )
            self.observation_spaces[self.possible_agents[agent_id]] = (
                gymnasium.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self._get_obs_agent(agent_id)),),
                    dtype=self.single_agent_env.observation_space.dtype,
                )
            )

    def _create_base_gym_env(
        self, scenario: str, agent_conf: str, render_mode: str, **kwargs
    ) -> gymnasium.envs.mujoco.mujoco_env.MujocoEnv:
        """Creates the single agent environments that is to be factorized."""
        # load the underlying single agent Gymnasium MuJoCo Environment in `self.single_agent_env`
        if scenario in _MUJOCO_GYM_ENVIROMENTS:
            return gymnasium.make(f"{scenario}-v5", **kwargs, render_mode=render_mode)
        elif scenario in ["ManySegmentAnt"]:
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            asset_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets",
                f"many_segment_ant_{n_segs}_segments.auto.xml",
            )
            many_segment_ant.gen_asset(n_segs=n_segs, asset_path=asset_path)
            single_agent_env = gymnasium.make(
                "Ant-v5", xml_file=asset_path, **kwargs, render_mode=render_mode
            )
            os.remove(asset_path)
            return single_agent_env
        elif scenario in ["ManySegmentSwimmer"]:
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            asset_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets",
                f"many_segment_swimmer_{n_segs}_segments.auto.xml",
            )
            many_segment_swimmer.gen_asset(n_segs=n_segs, asset_path=asset_path)
            single_agent_env = gymnasium.make(
                "Swimmer-v5", xml_file=asset_path, **kwargs, render_mode=render_mode
            )
            os.remove(asset_path)
            return single_agent_env
        elif scenario in ["CoupledHalfCheetah"]:
            return TimeLimit(CoupledHalfCheetahEnv(render_mode), max_episode_steps=1000)
        else:
            raise NotImplementedError("Custom env not implemented!")

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, str],
    ]:
        """Runs one timestep of the environment using the agents's actions.

        Note: if step is called after the agents have terminated/truncated the envrioment will continue to work as normal
        Args:
            actions:
                the actions of all agents

        Returns:
            see pettingzoo.utils.env.ParallelEnv.step() doc
        """
        _, reward_n, is_terminal_n, is_truncated_n, info_n = self.single_agent_env.step(
            self.map_local_actions_to_global_action(actions)
        )

        rewards, terminations, truncations, info = {}, {}, {}, {}
        observations = self._get_obs()
        for agents in self.possible_agents:
            rewards[agents] = reward_n
            terminations[agents] = is_terminal_n
            truncations[agents] = is_truncated_n
            info[agents] = info_n

        if is_terminal_n or is_truncated_n:
            self.agents = []

        return observations, rewards, terminations, truncations, info

    def map_local_actions_to_global_action(
        self, actions: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Maps multi agent actions into single agent action space.

        Args:
            action: An dict representing the action of each agent

        Returns:
            The action of the whole domain (is what eqivilent single agent action would be)

        Raises:
            AssertionError:
                If the Agent action factorization is badly defined (if an action is double defined or not defined at all)
        """
        assert self.single_agent_env.action_space.shape is not None
        global_action = (
            np.zeros((self.single_agent_env.action_space.shape[0],)) + np.nan
        )
        for agent_id, partition in enumerate(self.agent_action_partitions):
            for act_index, body_part in enumerate(partition):
                assert np.isnan(
                    global_action[body_part.act_ids]
                ), "FATAL: At least one gym_env action is doubly defined!"
                global_action[body_part.act_ids] = actions[
                    self.possible_agents[agent_id]
                ][act_index]

        assert not np.isnan(
            global_action
        ).any(), "FATAL: At least one gym_env action is undefined!"
        return global_action

    def map_global_action_to_local_actions(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Maps single agent action into multi agent action spaces.

        Args:
            action: An array representing the actions of the single agent for this domain

        Returns:
            A dictionary of actions to be performed by each agent

        Raises:
            AssertionError:
                If the Agent action factorization sizes are badly defined
        """
        local_actions = {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            local_actions[self.possible_agents[agent_id]] = np.array(
                [action[node.act_ids] for node in partition]
            )

        # assert sizes
        assert len(local_actions) == len(self.action_spaces)
        for agent in self.possible_agents:
            assert len(local_actions[agent]) == self.action_spaces[agent].shape[0]

        return local_actions

    def map_global_state_to_local_observations(
        self, global_state: np.ndarray[np.float64]
    ) -> dict[str, np.ndarray[np.float64]]:
        """Maps single agent observation into multi agent observation spaces.

        Args:
            global_state:
                the global_state (generated from MaMuJoCo.state())

        Returns:
            A dictionary of states that would be observed by each agent given the 'global_state'
        """
        assert (
            self.observation_factorization is not None
        ), "to map states the MuJoCo environment must have `observation_structure` member variable"
        global_state = np.array(global_state)

        local_observation = {}
        for agent, partition in self.observation_factorization.items():
            local_observation[agent] = global_state[partition]

        # assert sizes
        assert len(local_observation) == len(self.action_spaces)
        for agent in self.possible_agents:
            assert (
                len(local_observation[agent]) == self.observation_spaces[agent].shape[0]
            )

        return local_observation

    def map_local_observations_to_global_state(
        self, local_observation: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Maps multi agent observations into single agent observation space.

        Args:
            local_obserations:
                the local observation of each agents (generated from MaMuJoCo.step())

        Returns:
            the global observations that correspond to a single agent (what you would get with MaMuJoCo.state())
        """
        assert (
            self.observation_factorization is not None
        ), "to map states the MuJoCo environment must have `observation_structure` member variable"

        global_observation = (
            np.zeros((self.single_agent_env.observation_space.shape[0],)) + np.nan
        )

        for agent, partition in self.observation_factorization.items():
            for local_idx, global_idx in enumerate(partition):
                assert (
                    np.isnan(global_observation[global_idx])
                    or global_observation[global_idx]
                    == local_observation[agent][local_idx]
                ), "FATAL: At least one gym_env observation is doubly defined!"
                global_observation[global_idx] = local_observation[agent][local_idx]

        assert not np.isnan(
            global_observation
        ).any(), "FATAL: At least one gym_env observation is undefined, observations can not be mapped."
        return global_observation

    def create_observation_mapping(self) -> dict[str, np.ndarray[np.float64]]:
        """Creates a cache of the observation factorization.

        The cache is intended to be used with `map_global_state_to_local_observations` & `map_local_observations_to_global_state`.

        Returns:
            A cache that indexes global osbervations to local.
        """
        if not hasattr(self.single_agent_env.unwrapped, "observation_structure"):
            return None

        class data_struct:
            def __init__(self, qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext):
                self.qpos = qpos
                self.qvel = qvel
                self.cinert = cinert
                self.cvel = cvel
                self.qfrc_actuator = qfrc_actuator
                self.cfrc_ext = cfrc_ext

        obs_struct = self.single_agent_env.unwrapped.observation_structure
        qpos_end_index = obs_struct["qpos"]
        qvel_end_index = qpos_end_index + obs_struct["qvel"]
        cinert_end_index = qvel_end_index + obs_struct.get("cinert", 0)
        cvel_end_index = cinert_end_index + obs_struct.get("cvel", 0)
        qfrc_actuator_end_index = cvel_end_index + obs_struct.get("qfrc_actuator", 0)
        cfrc_ext_end_index = qfrc_actuator_end_index + obs_struct.get("cfrc_ext", 0)

        global_index = np.arange(self.single_agent_env.observation_space.shape[0])
        assert len(global_index) == cfrc_ext_end_index, "wrong indexing"

        mujoco_data = data_struct(
            qpos=np.concatenate(
                [
                    np.zeros(obs_struct["skipped_qpos"], dtype=np.int64),
                    global_index[0:qpos_end_index],
                ]
            ),
            qvel=np.array(global_index[qpos_end_index:qvel_end_index]),
            cinert=np.concatenate(
                [
                    np.zeros(10, dtype=np.int64),
                    global_index[qvel_end_index:cinert_end_index],
                ]
            ),
            cvel=np.concatenate(
                [
                    np.zeros(6, dtype=np.int64),
                    global_index[cinert_end_index:cvel_end_index],
                ]
            ),
            qfrc_actuator=np.concatenate(
                [
                    np.zeros(6, dtype=np.int64),
                    global_index[cvel_end_index:qfrc_actuator_end_index],
                ]
            ),
            cfrc_ext=np.concatenate(
                [
                    np.zeros(6, dtype=np.int64),
                    global_index[qfrc_actuator_end_index:cfrc_ext_end_index],
                ]
            ),
        )

        if len(mujoco_data.cinert) > 10:
            mujoco_data.cinert = np.reshape(
                mujoco_data.cinert, self.single_agent_env.unwrapped.data.cinert.shape
            )
        if len(mujoco_data.cvel) > 6:
            mujoco_data.cvel = np.reshape(
                mujoco_data.cvel, self.single_agent_env.unwrapped.data.cvel.shape
            )
        if len(mujoco_data.cfrc_ext) > 6:
            mujoco_data.cfrc_ext = np.reshape(
                mujoco_data.cfrc_ext,
                self.single_agent_env.unwrapped.data.cfrc_ext.shape,
            )

        assert len(self.single_agent_env.unwrapped.data.qpos.flat) == len(
            mujoco_data.qpos
        )
        assert len(self.single_agent_env.unwrapped.data.qvel.flat) == len(
            mujoco_data.qvel
        )

        local_index = {}
        for agent_id, agent in enumerate(self.possible_agents):
            local_index[agent] = self._get_obs_agent(agent_id, mujoco_data)
        return local_index

    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        """See [pettingzoo.utils.env.ParallelEnv.observation_space](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.observation_space)."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Box:
        """See [pettingzoo.utils.env.ParallelEnv.action_space](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.action_space)."""
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """See [pettingzoo.utils.env.ParallelEnv.state](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.state)."""
        return self.single_agent_env.unwrapped._get_obs()

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Returns: all agent's observations in a dict[str, ActionType]."""
        # dev NOTE: ignores `self.single_agent_env._get_obs()` and builds observations using obsk.build_obs()
        observations = {}
        for agent_id, agent in enumerate(self.possible_agents):
            observations[agent] = self._get_obs_agent(agent_id)
        return observations

    def _get_obs_agent(self, agent_id: int, data=None) -> np.ndarray:
        """Get the observation of single agent.

        Args:
            agent_id: The id in self.possible_agents.values()
            data: An optional overwrite of the MuJoCo data, defaults to the data at the current time step

        Returns:
            The observation of the agent given the data
        """
        if self.agent_obsk is None:
            return self.single_agent_env.unwrapped._get_obs()

        index_only = True
        if data is None:
            data = self.single_agent_env.unwrapped.data
            index_only = False

        return build_obs(
            data,
            self.k_dicts[agent_id],
            self.local_categories,
            self.mujoco_globals,
            self.global_categories,
            index_only,
        )

    def reset(self, seed: int | None = None, options: dict[str, any] | None = None):
        """Resets the the `single_agent_env`.

        Args:
            seed: see [pettingzoo.utils.env.ParallelEnv.reset()](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.reset) doc.
            options: passed to the single agent env's `reset`.

        Returns:
            Initial observations and info
        """
        _, info_n = self.single_agent_env.reset(seed=seed, options=options)
        info = {}
        for agent in self.possible_agents:
            info[agent] = info_n
        self.agents = self.possible_agents
        return self._get_obs(), info

    def render(self):
        """Renders the MuJoCo environment using the mechanism of the single agent Gymnasium-MuJoCo.

        Returns:
            The same return value as the single agent Gymnasium.MuJoCo
            see https://gymnasium.farama.org/environments/mujoco/
        """
        return self.single_agent_env.render()

    def close(self):
        """See [pettingzoo.utils.env.ParallelEnv.close](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.close)."""
        self.single_agent_env.close()

    def _generate_local_categories(self, scenario: str) -> list[list[str]]:
        """Generated the default observation categories for each observation depth.

        Args:
            scenario: the mujoco task

        Returns:
            a list of observetion types per observation depth
        """
        if self.agent_obsk is None:
            return [[]]

        if scenario in ["Ant", "ManySegmentAnt"]:
            k_categories = [["qpos", "qvel", "cfrc_ext"], ["qpos"]]
        elif scenario in ["Humanoid", "HumanoidStandup"]:
            k_categories = [
                ["qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"],
                ["qpos"],
            ]
        elif scenario in ["CoupledHalfCheetah"]:
            k_categories = [
                ["qpos", "qvel", "ten_J", "ten_length", "ten_velocity"],
                ["qpos"],
            ]
        elif scenario in ["Reacher"]:
            k_categories = [["qpos", "qvel", "fingertip_dist"], ["qpos"]]
        else:
            k_categories = [["qpos", "qvel"], ["qpos"]]

        # extend the length of categories to match `self.agent_obsk` by repeating the last element
        categories = [
            k_categories[k if k < len(k_categories) else -1]
            for k in range(self.agent_obsk + 1)
        ]
        return categories


# These are the export functions (for `PettingZoo` style exportations)
env = pettingzoo.utils.conversions.aec_wrapper_fn(MultiAgentMujocoEnv)
parallel_env = MultiAgentMujocoEnv
raw_parallel_env = MultiAgentMujocoEnv
