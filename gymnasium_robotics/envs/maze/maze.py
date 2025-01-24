import math
import tempfile
import xml.etree.ElementTree as ET
from os import path
from typing import Dict, List, Optional, Union

import numpy as np

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze.maps import COMBINED, GOAL, RESET, U_MAZE


class Maze:
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @property
    def maze_map(self) -> List[List[Union[str, int]]]:
        return self._maze_map

    @property
    def maze_size_scaling(self) -> float:
        return self._maze_size_scaling

    @property
    def maze_height(self) -> float:
        return self._maze_height

    @property
    def unique_goal_locations(self) -> List[np.ndarray]:
        return self._unique_goal_locations

    @property
    def unique_reset_locations(self) -> List[np.ndarray]:
        return self._unique_reset_locations

    @property
    def combined_locations(self) -> List[np.ndarray]:
        return self._combined_locations

    @property
    def map_length(self) -> int:
        return self._map_length

    @property
    def map_width(self) -> int:
        return self._map_width

    @property
    def x_map_center(self) -> float:
        return self._x_map_center

    @property
    def y_map_center(self) -> float:
        return self._y_map_center

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        i = math.floor((self.y_map_center - xy_pos[1]) / self.maze_size_scaling)
        j = math.floor((xy_pos[0] + self.x_map_center) / self.maze_size_scaling)
        return np.array([i, j])

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))

        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        # Make temporary file object and make the string path to our new file
        tmp_dir = tempfile.TemporaryDirectory()
        temp_xml_path = path.join(tmp_dir.name, "ant_maze.xml")

        # Write the new xml to the temporary file
        with open(temp_xml_path, "wb") as xml_file:
            tree.write(xml_file)

        return (
            maze,
            temp_xml_path,
            tmp_dir,  # The tmp_dir object is returned to keep it alive
        )


class MazeEnv(GoalEnv):
    def __init__(
        self,
        agent_xml_path: str,
        reward_type: str = "dense",
        continuing_task: bool = True,
        maze_map: List[List[Union[int, str]]] = U_MAZE,
        maze_size_scaling: float = 1.0,
        maze_height: float = 0.5,
        position_noise_range: float = 0.25,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.maze, self.tmp_xml_file_path, self.tmp_dir = Maze.make_maze(
            agent_xml_path, maze_map, maze_size_scaling, maze_height
        )

        self.position_noise_range = position_noise_range

    def generate_target_goal(self) -> np.ndarray:
        assert len(self.maze.unique_goal_locations) > 0
        goal_index = self.np_random.integers(
            low=0, high=len(self.maze.unique_goal_locations)
        )
        goal = self.maze.unique_goal_locations[goal_index].copy()
        return goal

    def generate_reset_pos(self) -> np.ndarray:
        assert len(self.maze.unique_reset_locations) > 0

        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        while np.linalg.norm(reset_pos - self.goal) <= 0.5:
            reset_index = self.np_random.integers(
                low=0, high=len(self.maze.unique_reset_locations)
            )
            reset_pos = self.maze.unique_reset_locations[reset_index].copy()

        return reset_pos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        super().reset(seed=seed)

        if options is None:
            goal = self.generate_target_goal()
            # Add noise to goal position
            self.goal = self.add_xy_position_noise(goal)
            reset_pos = self.generate_reset_pos()
        else:
            if "goal_cell" in options and options["goal_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["goal_cell"][0]
                assert self.maze.map_width > options["goal_cell"][1]
                assert (
                    self.maze.maze_map[options["goal_cell"][0]][options["goal_cell"][1]]
                    != 1
                ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

                goal = self.maze.cell_rowcol_to_xy(options["goal_cell"])
            else:
                goal = self.generate_target_goal()

            # Add noise to goal position
            self.goal = self.add_xy_position_noise(goal)

            if "reset_cell" in options and options["reset_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["reset_cell"][0]
                assert self.maze.map_width > options["reset_cell"][1]
                assert (
                    self.maze.maze_map[options["reset_cell"][0]][
                        options["reset_cell"][1]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        # Add noise to reset position
        self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
        noise_x = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        noise_y = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        xy_pos[0] += noise_x
        xy_pos[1] += noise_y

        return xy_pos

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            return (distance <= 0.45).astype(np.float64)

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            if (
                bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
                and len(self.maze.unique_goal_locations) > 1
            ):
                # Generate another goal
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)
                # Update the position of the target site for visualization
                self.update_target_site_pos()

            return False

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        return False

    def update_target_site_pos(self, pos):
        raise NotImplementedError

    def __del__(self):
        self.tmp_dir.cleanup()
        super().__del__()
