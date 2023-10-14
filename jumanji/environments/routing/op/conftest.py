import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.op.env import OP
from jumanji.environments.routing.op.generator import Generator, ProportionalGenerator
from jumanji.environments.routing.op.reward import DenseReward, SparseReward
from jumanji.environments.routing.op.types import State


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def op_dense_reward(dense_reward: DenseReward) -> OP:
    """Instatiates an OP environment with dense reward, 10 nodes and
    length budget of 5.
    """
    return OP(generator=ProportionalGenerator(num_nodes=10, max_length=5),
              reward_fn=dense_reward)


@pytest.fixture
def op_sparse_reward(sparse_reward: SparseReward) -> OP:
    """Instatiates an OP environment with sparse reward, 10 nodes and
    length budget of 5.
    """
    return OP(generator=ProportionalGenerator(num_nodes=10, max_length=5),
              reward_fn=sparse_reward)


class DummyGenerator(Generator):
    """Hardcoded 'Generator' mainly used for the testing and debugging. It
    deterministically outputs a hardcoded instance with 4 nodes and maximum
    length budget of 0.4.
    """

    def __init__(self) -> None:
        super().__init__(num_nodes=4, max_length=0.4)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns an
        orienteering problem without any visited nodes and starting at the depot


        Args:
            keys: jax random key for any stochasticity used in the generation process.
            Not used in this instance generator.

        Returns:
            A OP State
        """
        coordinates = jnp.array(
            [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            float,
        )

        # Initially, the position is set at the depot.
        position = jnp.array(0, jnp.int32)
        num_visited = jnp.array(1, jnp.int32)

        # The prize and length at depot is 0
        length = jnp.array([0, 0.07, 0.07, 0.11, 0.15])
        prizes = self._generate_prizes(key, length)

        # The initial travel budget (max length)
        remaining_budget = jnp.array(0.4, jnp.int32)

        # Initially the agent has only visited the depot.
        visited_mask = jnp.array([True, False, False, False, False], bool)
        trajectory = jnp.array([0, 0, 0, 0, 0], jnp.int32)

        del key

        state = State(
            coordinates=coordinates,
            position=position,
            num_visited=num_visited,
            prizes=prizes,
            length=length,
            remaining_budget=remaining_budget,
            visited_mask=visited_mask,
            trajectory=trajectory,
            key=jax.random.PRNGKey(0),
        )

        return state

    def _generate_prizes(self, _key: chex.PRNGKey, _length: chex.Array
                         ) -> chex.Array:
        """An abstract method responsible for generating an instance of node prizes
        based on the variant of the orienteering problem.

        Args:
            length: the distance between the depot and prized nodes

        Returns:
            prizes: the prizes associated with nodes to be visited.
        """
        prizes = jnp.array([0, 0.07, 0.08, 0.11, 0.15])

        return prizes
