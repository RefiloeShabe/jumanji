import jax
import jax.numpy as jnp

from jumanji.environments.routing.op.env import OP
from jumanji.environments.routing.op.reward import DenseReward, SparseReward


def test_dense_reward(op_dense_reward: OP, dense_reward: DenseReward) -> None:
    dense_reward = jax.jit(dense_reward)
    step_fn = jax.jit(op_dense_reward.step)
    state, timestep = op_dense_reward.reset(jax.random.PRNGKey(0))

    # check that the reward is correct for any node
    state, timestep = step_fn(state, 0)
    for action in range(1, op_dense_reward.num_nodes):
        next_state, _ = step_fn(state, action)
        node_prize = state.prizes[action]
        reward = dense_reward(state, action, next_state, is_valid=True)
        assert reward == node_prize

    # check the reward for invalid action
    next_state, _ = step_fn(state, 0)
    penalty = -jnp.sqrt(2) * (op_dense_reward.num_nodes + 1)
    reward = dense_reward(state, 0, next_state, is_valid=False)
    assert reward == penalty


def test_sparse_reward(op_sparse_reward: OP, sparse_reward: SparseReward) -> None:
    sparse_reward = jax.jit(sparse_reward)
    step_fn = jax.jit(op_sparse_reward.step)
    state, timestep = op_sparse_reward.reset(jax.random.PRNGKey(0))
    penalty = -jnp.sqrt(2) * (op_sparse_reward.num_nodes + 1)

    # check that all but the last step leads to 0 reward
    next_state = state
    while not timestep.last():
        for action, is_valid in enumerate(timestep.observation.action_mask):
            if is_valid:
                next_state, timestep = step_fn(state, action)
                reward = sparse_reward(state, action, next_state, is_valid)
                if timestep.last():
                    # at the end of the episode, check that the reward is the total
                    # prizes of visited nodes
                    total_prize = jnp.sum(state.prizes, where=next_state.visited_mask)
                    print(action, reward, next_state.visited_mask)
                    assert reward == total_prize
                else:
                    # check that the reward is 0 for non-final valid action
                    reward = sparse_reward(state, action, next_state, is_valid)
                    assert reward == 0

            else:
                # check that a penalty is given for invalid action
                invalid_next_state, _ = step_fn(state, action)
                reward = sparse_reward(state, action, invalid_next_state, is_valid)
                assert reward == penalty
        state = next_state
