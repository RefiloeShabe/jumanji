import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.op.env import OP 
from jumanji.environments.routing.op.types import State
from jumanji.environments.routing.op.constants import DEPOT_IDX
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


class TestSparseOP:
    def test_op_sparse_reset(self, op_sparse_reward: OP) -> None:
        """Validates the jitted reset of the environment"""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(op_sparse_reward.reset, n=1))
        
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Check that it does not compile twice
        state, timestep = reset_fn(key)
        
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        
        # Initially the position is at depot; the current remaining travel length is max length
        assert state.remaining_max_length == op_sparse_reward.max_length > 0
        # Length from depot to depot is 0
        assert state.length[DEPOT_IDX] == 0
        # The depot is initially visited
        assert state.visited_mask[DEPOT_IDX]
        assert state.visited_mask.sum() == 1
        # First visited position is the depot
        assert state.trajectory[0] == DEPOT_IDX
        assert state.num_visited == 1
        # Prize of the depot
        assert state.prizes[DEPOT_IDX] == 0
        
        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # reset function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(state) 
        
    def test_op_sparse_step(self, op_sparse_reward: OP) -> None:
        """Validates the jitted step of the environment"""
        chex.clear_trace_counter()
        
        step_fn = chex.assert_max_traces(op_sparse_reward.step, n=1)
        step_fn = jax.jit(step_fn)
        
        key = jax.random.PRNGKey(0)
        state, timestep = op_sparse_reward.reset(key)
        
        # Starting position is the depot, first action to visit first node
        new_action = 1
        new_state, next_timestep = step_fn(state, new_action)
        
        # Check that the state has changed
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.remaining_max_length, state.remaining_max_length)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.num_visited, state.num_visited)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        
        assert_is_jax_array_tree(new_state)
        
        # Check that the state was correctly changed
        assert new_state.visited_mask[new_action]
        assert new_state.visited_mask.sum() == 1
        
        # Check that the state does not change when taking the same action again
        state = new_state
        new_state, next_timestep = step_fn(state, new_action)
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.remaining_max_length, state.remaining_max_length)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.num_visited, state.num_visited)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
             
    def test_op_sparse_reward__does_not_smoke(self, op_sparse_reward: OP) -> None:
        """Tests that we can run an episode without any errors"""
        check_env_does_not_smoke(op_sparse_reward)  
        
    def test_op_sparse__trajectory_action(self, op_sparse_reward: OP) -> None:
        """Checks that the agent goes back to the depot when the remaining travel budget 
        does not allow visiting extra nodes and that appropriate reward is recieved.
        """
        step_fn = jax.jit(op_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = op_sparse_reward.reset(key)
        
        while not timestep.last():
            # Check that the budget remains positive
            assert state.remaining_max_length > 0
            
            # Check that the reward is 0 while trajectory is not done
            assert timestep.reward == 0
            
            action = jnp.argmax(timestep.observation.action_mask)
            state, timestep = step_fn(state, action)
            
        # Check that the reward is positive when the trajectory is done
        assert timestep.reward > 0
        
        # Check that no action can be taken: the remainining nodes are too far
        assert not jnp.any(timestep.observation.action_mask)
        assert timestep.last()  
        
    def test_op_sparse_invalid_action(self, op_sparse_reward: OP) -> None:
        """ Checks that an invalid action leads to a termination and the appropriate reward is
        received.
        """
        step_fn = jax.jit(op_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = op_sparse_reward.reset(key)
        
        first_position = state.position
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % op_sparse_reward.num_nodes
        )
        
        for a in actions:
            assert timestep.reward == 0
            assert not timestep.last()
            state, timestep = step_fn(state, a)
            
        # Last action is invalid because it was already taken
        assert timestep.reward < 0
        assert timestep.last()       
          
             
class TestDenseOP:
    def test_op_dense_reset(self, op_dense_reward: OP) -> None:
        """Validates the jitted reset of the environment"""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(op_dense_reward.reset, n=1))
        
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Check that it does not compile twice
        state, timestep = reset_fn(key)
        
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        
        # Initially the position is at depot; the current remaining travel budget is max length
        assert state.remaining_max_length == op_dense_reward.max_length > 0
        # Length from depot to depot is 0
        assert state.length[DEPOT_IDX] == 0
        # The depot is initially visited
        assert state.visited_mask[DEPOT_IDX]
        assert state.visited_mask.sum() == 1
        # First visited position is the depot
        assert state.trajectory[0] == DEPOT_IDX
        assert state.num_visited == 1
        # Prize of the depot
        assert state.prizes[DEPOT_IDX] == 0

        assert_is_jax_array_tree(state) 
        
    def test_op_dense_step(self, op_dense_reward: OP) -> None:
        """Validates the jitted step of the environment"""
        chex.clear_trace_counter()
        
        step_fn = chex.assert_max_traces(op_dense_reward.step, n=1)
        step_fn = jax.jit(step_fn)
        
        key = jax.random.PRNGKey(0)
        state, timestep = op_dense_reward.reset(key)
        
        # Starting position is the depot, first action to visit first node
        new_action = 1
        new_state, next_timestep = step_fn(state, new_action)
        
        # Check that the state has changed
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.remaining_max_length, state.remaining_max_length)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.num_visited, state.num_visited)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        
        assert_is_jax_array_tree(new_state)
        
        # Check that the state was correctly changed
        assert new_state.visited_mask[new_action]
        assert new_state.visited_mask.sum() == 1
        
        # New step with same action should be invalid
        state = new_state
        new_state, next_timestep = step_fn(state, new_action)
        
        # Check that the state does not change when taking the same action again
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.remaining_max_length, state.remaining_max_length)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.num_visited, state.num_visited)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
            
    def test_op_dense_reward__does_not_smoke(self, op_dense_reward: OP) -> None:
        """Tests that we can run an episode without any errors"""
        check_env_does_not_smoke(op_dense_reward)  
        
    def test_op_dense__trajectory_action(self, op_dense_reward: OP) -> None:
        """Tests that the agent goes back to the depot when the remaining travel budget 
        does not allow visiting extra nodes and that appropriate reward is recieved.
        """
        step_fn = jax.jit(op_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = op_dense_reward.reset(key)
        
        while not timestep.last():
            # Check that the budget remains positive
            assert state.remaining_max_length > 0
            
            # Check that the reward is positive at each but first step
            assert timestep.reward > 0 or timestep.first()
            
            action = jnp.argmax(timestep.observation.action_mask)
            state, timestep = step_fn(state, action)
            
        # Check that the reward is positive when the trajectory is done
        assert timestep.reward > 0
        
        # Check that no action can be taken: the remainining nodes are too far
        assert not jnp.any(timestep.observation.action_mask)
        assert timestep.last()  
        
    def test_op_dense_invalid_action(self, op_dense_reward: OP) -> None:
        """ Checks that an invalid action leads to a termination and that the appropriate reward 
        is received.
        """
        step_fn = jax.jit(op_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = op_dense_reward.reset(key)
        
        first_position = state.position
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % op_dense_reward.num_nodes
        )
        
        for a in actions:
            assert not timestep.last()
            state, timestep = step_fn(state, a)
            assert timestep.reward > 0
            
        # Last action is invalid because it was already taken
        assert timestep.reward < 0
        assert timestep.last()         
        
def test_op__equivalence_dense_sparse_reward(
    op_dense_reward: OP, op_sparse_reward: OP
) -> None:
    """ Checks that both dense and sparse environments return equal rewards
    when an episode is done
    """
    dense_step_fn = jax.jit(op_dense_reward.step)
    sparse_step_fn = jax.jit(op_sparse_reward.step)
    key = jax.random.PRNGKey(0)

    # Dense reward
    state, timestep = op_dense_reward.reset(key)
    return_dense = timestep.reward
    while not timestep.last():
        state, timestep = dense_step_fn(state, jnp.argmin(state.visited_mask))
        return_dense += timestep.reward

    # Sparse reward
    state, timestep = op_sparse_reward.reset(key)
    return_sparse = timestep.reward
    while not timestep.last():
        state, timestep = sparse_step_fn(state, jnp.argmin(state.visited_mask))
        return_sparse += timestep.reward

    # Check that both returns are the same and not the invalid action penalty
    assert (return_sparse == return_dense > -2 * op_dense_reward.num_nodes * jnp.sqrt(2))              