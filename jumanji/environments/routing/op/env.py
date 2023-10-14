from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp


from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from jumanji.environments.routing.op.constants import DEPOT_IDX
from jumanji.environments.routing.op.generator import Generator, UniformGenerator
from jumanji.environments.routing.op.types import State, Observation
from jumanji.environments.routing.op.reward import DenseReward, RewardFn


class OP(Environment[State]):
    """ Orienteering Problem (OP) as described in [1].

    - observation: Observation
        - coordinates: jax array (float) of shape (num_nodes + 1, 2)
            the coordinates of each node.
        - position: jax array (int32) of shape ()
            the indec corresponding to the last visited node
        - trajectory: jax array (int32) of shape (num_nodes + 1, )
            array of node indicies defining the route (set to DEPOT_IDX if not filled
            yet)
        - action_mask: jax array (bool) of shape (num_nodes + 1, )
            binary mask (False/True <--> illegal/legal <--> can/cannot be visited)
        - prizes: jax array (float) of shape (num_nodes + 1, ), could be either:
            - constant: all nodes have the same prize equal to 1.
            - uniform: the prizes of the nodes are randomly sampled from a uniform
            distribution on a unit square.
            - proportional: the prizes of the nodes are proportional to the length
            between the depot and the nodes.
        - length: jax array (float) of shape (num_nodes + 1, )
            the length between each node and the depot (0.0 for the depot)


    - action: jax array (int32) of shape ()
        [0, ..., num_nodes] -> nodes to visit. 0 corresponding to visiting the depot.


    - reward: jax array (float) of shape (), could be either:
        - dense: the prize associated with the chosen next node to go to.
            It is 0 for the first chosen node and the last node.
        - sparse: the total prize collect at the end of the episode. The total
            prize is defined as the sum of prizes associated with visited nodes in an
            episode.
            It is computed by starting at the first node and ending there, visiting a
            subset of all nodes.
        In both cases, the reward is zero is the action is invalid, i.e. a previsouly
        selected node is selected again or it is too far to make it to the depot in
        given time.


    - episode termination:
        - if no action can be performed, i.e. the remaining time budget is zero.
        - if an invalid action is taken, i.e. an already visited is chosen or
            a chosen node is too far to reach the depot within the time budget.


    - state: 'State'
        - coordinates: jax array (float) of shape (num_nodes + 1, 2)
            the coordinates of each node.
        - position: jax array (int32) of shape ()
            the index corresponding to the last visited node.
        - visited_mask: jax array (bool) of shape (num_nodes + 1, )
            binary mask (False/True <--> not visited/visited).
        - trajectory: jax array (int32) of shape (num_nodes, )
            array of node indicies defining the route (set to DEPOT_IDX if not filled
            yet).
        - num_visited: int32
                number of total nodes visited
        - prizes: jax array (float) of shape (num_nodes + 1, )
            the associated prizez of each node and the depot note (0.0 for the depot)
        - length: jax array (float) of shape (num_nodes + 1, )
            the length between each node and the depot (0.0 for the depot)
        -remaining_budget: jax array (float) of shape ()
            the remaining length budget


    [1] Wouter Kool, Herke van Hoof, and Max Welling. (2019). "Attention, learn to
    solve routing problems!"

    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiates an OP environment.


        Args:
            generator: 'Generator' whose '__call__' method instantiates an environment
            instance. The default option is 'UniformGenerator' which randomly generates
            OP instances with 20 nodes and prizes sampled from a uniform distribution.
            reward_fn: RewardFn whose '__call__' method computes the reward of an
            environment transition. The function must compute the reward based on the
            current state, the chosen action and the next state. Implement options are
            ['DenseReward', 'SparseReward']. Defaults to 'DenseReward'.
            viewer: 'Viewer' used for rendering. Defaults to the 'OPViewer' with 'human'
            render mode.
        """

        self.generator = generator or UniformGenerator(
            num_nodes=20,
            max_length=2,
        )
        self.num_nodes = self.generator.num_nodes
        self.max_length = self.generator.max_length
        self.reward_fn = reward_fn or DenseReward()
        self._viewer = viewer

    def __repr__(self) -> str:
        return f"OP environment with {self.num_nodes} nodes and travel budget of {self.max_length}."

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment.

        Args:
            Key: used to randomly generate the coordinates.


        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: Timestep object corresponding to the first timestep returned
            by the environment.
        """
        state = self.generator(key)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: chex.Numeric
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.


        Args:
            state: 'State' object containing the dynamics of the environment.
            action: 'Array' containing the index of the next node to visit.


        Returns:
            state: the next state of the environment.
            timestep: the timestep to be observed.    
        """

        travelled_distance = jnp.linalg.norm(state.coordinates[state.position] - state.coordinates[action])
        valid_length = state.remaining_budget - travelled_distance
        is_valid = ~state.visited_mask[action] & (state.length[action] <= valid_length)

        next_state = jax.lax.cond(
            is_valid,
            self._update_state,
            lambda *_: state,
            state,
            action,
        )

        reward = self.reward_fn(state, action, next_state, is_valid)

        observation = self._state_to_observation(next_state)

        # Terminate if visit depot or the action is invalid or there are no nodes to visit 
        is_done = next_state.visited_mask[DEPOT_IDX] | ~is_valid | next_state.visited_mask.all()

        # is_done = next_state.visited_mask.all() | ~is_valid

        timestep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """"Returns the observation spec.

        Returns:
            Spec for the 'Observation' whose fields are:
            - coordinates: BoundedArray (float) of shape (num_nodes + 1, ).
            - position: DiscreteArray (int32) of shape (num_nodes + 1, ).
            - trajectory: BoundedArray (int32) of shape (num_nodes, ).
            - prizes: BoundedArray (float) of shape (num_nodes + 1, ).
            - length: BoundedArray (float) of shape (num_nodes + 1, ).
            - action_mask: BoundedArray (bool) of shape (num_nodes + 1, )
        """
        coordinates = specs.BoundedArray(
            shape=(self.num_nodes + 1, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="coordinates",
        )
        position = specs.DiscreteArray(
            self.num_nodes + 1, dtype=jnp.int32, name="position"
        )
        trajectory = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_nodes + 1,
            name="trajectory",
        )
        prizes = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            minimum=0.0,
            maximum=self.max_length,
            dtype=float,
            name="prizes",
        )
        length = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            minimum=0.0,
            maximum=self.max_length,
            dtype=float,
            name="length"
        )
        remaining_budget = specs.BoundedArray(
            shape=(), minimum=0.0, maximum=self.max_length, dtype=float, name="remaining_budget"
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            coordinates=coordinates,
            position=position,
            trajectory=trajectory,
            prizes=prizes,
            length=length,
            remaining_budget=remaining_budget,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """"Returns the action spec.

        Returns:
            action_spec: a 'specs.DiscreteArray' array.
        """

        return specs.DiscreteArray(self.num_nodes + 1, name="action")

    def _update_state(self, state: State, action: chex.Numeric) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            action: int32, index of the next position to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """

        # compute traveled distance: distance between previous node and currrent node
        travelled_distance = jnp.linalg.norm(state.coordinates[state.position] - state.coordinates[action])

        # Set depot to False (valid to visit) since it can be visited again
        visited_mask = state.visited_mask.at[DEPOT_IDX].set(False)

        return State(
            coordinates=state.coordinates,
            position=action,
            visited_mask=visited_mask.at[action].set(True),
            trajectory=state.trajectory.at[state.num_visited].set(action),
            num_visited=state.num_visited + 1,
            prizes=state.prizes,
            length=state.length,
            remaining_budget=state.remaining_budget - travelled_distance,
            key=state.key,

        )

    def _action_mask(self, state: State) -> chex.Array:
        """Defines a mask for actions that are not valid"""

        # Calculate distances from the current position to all other coordinates
        distances = Generator._distance_between_two_nodes(state.coordinates[state.position], state.coordinates)
        valid_lengths = state.remaining_budget - distances
        action_mask = (~state.visited_mask) & (state.length <= valid_lengths)

        # The depot is reachable if we are not at it already.
        action_mask = action_mask.at[DEPOT_IDX].set(state.position != DEPOT_IDX)

        return action_mask

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state to an observation.

        Args:
        state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        action_mask = self._action_mask(state)

        return Observation(
            coordinates=state.coordinates,
            position=state.position,
            trajectory=state.trajectory,
            prizes=state.prizes,
            length=state.length,
            remaining_budget=state.remaining_budget,
            action_mask=action_mask,
        )
