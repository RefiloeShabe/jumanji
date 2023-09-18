import chex
import jax
import jax.numpy as jnp
import pytest

import env, generator, reward, types_


@pytest.fixture
def dense_reward() -> reward.DenseReward:
    return reward.DenseReward()


@pytest.fixture
def sparse_reward() -> reward.SparseReward:
    return reward.SparseReward()


@pytest.fixture
def op_dense_reward(dense_reward: reward.DenseReward) -> env.OP:
    """Instatiates an OP environment with dense reward, 4 nodes and length budget of 2.
    """
    return env.OP(
        generator=generator.UniformGenerator(num_nodes=4, max_length=2),
        reward_fn=reward.DenseReward,
    )
    
    
@pytest.fixture
def op_sparse_reward(sparse_reward: reward.SparseReward) -> env.OP:
    """Instatiates an OP environment with sparse reward, 4 nodes and length budget of 2.
    """  
    return env.OP(
        generator=generator.UniformGenerator(num_nodes=4, max_length=2),
        reward_fn=reward.SparseReward,
    )  


class DummyGenerator(generator.Generator):
    """Hardcoded 'Generator' mainly used for the testing and debugging. It deterministically outputs a
    hardcoded instance with 4 nodes and maximum length budget of 2.
    """
    
    def __init__(self) -> None:
        super().__init__(num_nodes=4, max_length=2)
        
    def __call__(self, key: chex.PRNGKey) -> types_.State:
        """Call method responsible for generating a new state. It returns an orienteering problem without
        any visited nodes and starting at the depot
        
        
        Args:
            keys: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.
                
        Returns:
            A OP State        
        """
        del key
        
        coordinates = jnp.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
            float,
        )
        
        # Initially, the position is set at the depot.
        position = jnp.array(0, jnp.int32)
        num_visited = jnp.array(1, jnp.int32)
        
        # The prize and length at depot is 0
        prizes = jnp.array([0, 1, 0.7, 0.2, 1])
        length = jnp.array([0, 0.5, 0.5, 0.8, 0.2])
        
        # The initial travel budget (max length)
        remaining_max_length = jnp.array(2, jnp.int32)
        
        # Initially the agent has only visited the depot.
        visited_mask = jnp.array([True, False, False, False, False], bool)
        trajectory = jnp.array([0, 0, 0, 0, 0], jnp.int32)
        
        state = types_.State(
            coordinates=coordinates,
            position=position,
            num_visited=num_visited,
            prizes=prizes,
            length=length,
            remaining_max_length=remaining_max_length,
            visited_mask=visited_mask,
            trajectory=trajectory,
            key=jax.random.PRNGKey(0),
        )
        return state    