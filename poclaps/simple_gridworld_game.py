from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.environments import spaces


def stringify_grid(grid_size: int, symbols_on_grid: dict):
    result = "-" * (grid_size * 4 + 1) + "\n"
    for y in range(grid_size):
        result += "|"
        for x in range(grid_size):
            symbol = symbols_on_grid.get((x, y), " ")
            result += f" {symbol} |"
        result += "\n" + "-" * (grid_size * 4 + 1) + "\n"
    return result


def print_grid(grid_size: int, symbols_on_grid: dict):
    print(stringify_grid(grid_size, symbols_on_grid))


@struct.dataclass
class EnvState:
    goal_pos: jnp.ndarray
    agent_pos: jnp.ndarray
    time: int = 0


@struct.dataclass
class EnvParams(environment.EnvParams):
    grid_size: int = 5
    max_steps_in_episode: int = 20


def observation_fn(params: EnvParams, env_state: EnvState) -> jnp.ndarray:
    # agent_pos_x_1h = jax.nn.one_hot(env_state.agent_pos[X], 5).flatten()
    # agent_pos_y_1h = jax.nn.one_hot(env_state.agent_pos[Y], 5).flatten()
    # goal_pos_x_1h = jax.nn.one_hot(env_state.goal_pos[X], 5).flatten()
    # goal_pos_y_1h = jax.nn.one_hot(env_state.goal_pos[Y], 5).flatten()

    # return jnp.concatenate([
    #     agent_pos_x_1h,
    #     agent_pos_y_1h,
    #     goal_pos_x_1h,
    #     goal_pos_y_1h
    # ], dtype=jnp.float32)
    return jnp.array([
        env_state.agent_pos / params.grid_size,
        env_state.goal_pos / params.grid_size,
    ], dtype=jnp.float32)


class GridAction:
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    N_ACTIONS = 5
    ACTIONS = [
        NOOP, UP, DOWN, LEFT, RIGHT
    ]


X = 0
Y = 1


class SimpleGridWorldGame(environment.Environment):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        k1, k2 = jax.random.split(key)
        goal_pos = jax.random.randint(k1, (2,), 0, params.grid_size)
        agent_pos = jax.random.randint(k2, (2,), 0, params.grid_size)
        env_state = EnvState(goal_pos, agent_pos)
        return observation_fn(params, env_state), env_state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return observation_fn(params or self.default_params, state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        reached_goal = jnp.all(state.agent_pos == state.goal_pos)

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(reached_goal, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleGridWorld-v0"
    
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return GridAction.N_ACTIONS

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # return spaces.Box(0, 1, (4 * params.grid_size,), dtype=jnp.float32)
        return spaces.Box(0, 1, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                'agent_pos': spaces.Box(0, (params.grid_size - 1), (2,), dtype=jnp.float32),
                'goal_pos': spaces.Box(0, (params.grid_size - 1), (2,), dtype=jnp.float32),
                'time': spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        
        UP_DELTA = jnp.array([0, -1])
        DOWN_DELTA = jnp.array([0, 1])
        LEFT_DELTA = jnp.array([-1, 0])
        RIGHT_DELTA = jnp.array([1, 0])

        up_mask = action == GridAction.UP
        down_mask = action == GridAction.DOWN
        left_mask = action == GridAction.LEFT
        right_mask = action == GridAction.RIGHT

        new_agent_pos = state.agent_pos + (
            up_mask * UP_DELTA +
            down_mask * DOWN_DELTA +
            left_mask * LEFT_DELTA +
            right_mask * RIGHT_DELTA
        )
        new_agent_pos = jnp.clip(new_agent_pos, 0, params.grid_size - 1)
    
        # Update state dict and evaluate termination conditions
        state = state.replace(
            agent_pos=new_agent_pos,
            time=state.time + 1
        )
        done = self.is_terminal(state, params)

        reward = done * 2.0 - 1.0

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def to_string(self, state: EnvState, params: EnvParams) -> str:
        symbols_on_grid = {
            tuple(state.goal_pos.tolist()): "G",
            tuple(state.agent_pos.tolist()): "A"
        }
        return stringify_grid(params.grid_size, symbols_on_grid)
