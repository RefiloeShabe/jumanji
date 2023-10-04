import abc
import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.op.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action, the next
        state and whether the action is valid.
        """


class SparseReward(RewardFn):
    """The total prize collected at the end of the episode. The total prize is defined
    as the sum of the prizes from each visited node within the time constraint.
    The reward is 0 if the episode is not terminated yet.
    It is `-2 * num_nodes * sqrt(2)` if the action is invalid, i.e a node that was
    previously selected is selected again or the return to the depot is impossible
    within the time budget.
    """
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
    ) -> chex.Numeric:
        compute_sparse_reward = lambda: jax.lax.cond(
            is_valid,
            jnp.dot,
            lambda *_: jnp.array(-len(state.trajectory) * jnp.sqrt(2), float),
            next_state.visited_mask,
            next_state.prizes,
        )

        # If the episode is done
        is_done = next_state.visited_mask.all() | ~is_valid
        reward = jax.lax.cond(
            is_done,
            compute_sparse_reward,
            lambda: jnp.array(0, float),
        )
        return reward


class DenseReward(RewardFn):
    """The prize associated with the chosen node at the current timestep. The reward is
    `-2 * num_nodes * sqrt(2)` if the chosen action is invalid., i.e, a node that was
    previously visited is visited again or returning to the depot from the node is not
    possible within the time constraint.
    """
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
    ) -> chex.Numeric:
        reward = jax.lax.select(
            is_valid,
            state.prizes[action],
            jnp.array(-len(state.trajectory) * jnp.sqrt(2), float),
        )
        return reward
