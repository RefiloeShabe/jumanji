from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from chex import PRNGKey
from numpy.typing import NDArray


import jumanji
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
import generator, types_, reward, constants


class OP(Environment[types_.State]):
    """ Orienteering Problem (OP) as described in [1].
    
    - observation: Observation
        - coordinates: jax array (float) of shape (num_nodes + 1, 2)
            the coordinates of each node.
        - position: jax array (int32) of shape ()
            the indec corresponding to the last visited node 
        - trajectory: jax array (int32) of shape (num_nodes, )
            array of node indicies defining the route (set to DEPOT_IDX if not filled yet)
        - action_mask: jax array (bool) of shape (num_nodes + 1, )
            binary mask (False/True <--> illegal/legal <--> can/cannot be visited)
        - prizes: jax array (float) of shape (num_nodes + 1, ) 
            the associated prizez of each node and the depot note (0.0 for the depot)
        - length: jax array (float) of shape (num_nodes + 1, )
            the length between each node and the depot (0.0 for the depot)     
            
    
    - action: jax array (int32) of shape ()
        [0, ..., num_nodes] -> nodes to visit. 0 corresponding to visiting the depot. 
         
        
    - reward: jax array (float) of shape (), could be either:
        - dense: the prize associated with the chosen next node to go to.
            It is 0 for the first chosen node and the last node. 
        - sparse: the total prize collect at the end of the episode. The total 
            prize is defined as the sum of prizes associated with visited nodes in a episode.
            It is computed by starting at the first node and ending there, visiting a subset 
            of all nodes.
        In both cases, the reward is zero is the action is invalid, i.e. a previsouly selected 
        node is selected again or it is too far to make it to the depot in given time.
        
        
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
            array of node indicies defining the route (set to DEPOT_IDX if not filled yet).
        - num_visited: int32
                number of total nodes visited
        - prizes: jax array (float) of shape (num_nodes + 1, ) 
            the associated prizez of each node and the depot note (0.0 for the depot)
        - length: jax array (float) of shape (num_nodes + 1, )
            the length between each node and the depot (0.0 for the depot)  
        -remaining_max_length: jax array (float) of shape ()
            the remaining length budget  
            
    
    [1] Wouter Kool, Herke van Hoof, and Max Welling. (2019). "Attention, learn to solve routing problems!" 

    """
    
    
    def __init__(
        self,
        generator: Optional[generator.Generator] = None,
        reward_fn: Optional[reward.RewardFn] = None,
        viewer: Optional[Viewer[types_.State]] = None,
    ):
        """Instantiates an OP environment.
        
        
        Args:
            generator: 'Generator' whose '__call__' method instantiates an environment instance.
                The default option is 'UniformGenerator' which randomly generates 
                OP instances with 20 nodes sampled from a uniform distribution.
            reward_fn: RewardFn whose '__call__' method computes the reward of an environment
                transition. The function must compute the reward based on the current state,
                the chosen action and the next state.
                Implement options are ['DenseReward', 'SparseReward']. Defaults to 'DenseReward'.
            viewer: 'Viewer' used for rendering. Defaults to the 'OPViewer' with 'human' render mode.   
        """
        
        
        self.generator = generator or generator.UniformGenerator(
            num_nodes=20,
            max_length=2,
        )
        self.num_nodes = self.generator.num_nodes
        self.max_length = self.generator.max_length
        self.reward_fn = reward_fn or reward.DenseReward()
        self._viewer = viewer #or OPViewer(name="OP", render_mode="human")
        
        
    def __repr__(self) -> str:
        return f"OP environment with {self.num_nodes} nodes and travel budget of {self.max_length}."
    
    
    def reset(self, key: chex.PRNGKey) -> Tuple[types_.State, TimeStep[types_.Observation]]:
           """Reset the environment.
           
           Args:
                Key: used to randomly generate the coordinates.
                
                
           Returns:
                state: State object corresponding to the new state of the environment.
                timestep: Timestep object corresponding to the first timestep returned 
                by the environment.     
           """
           state = self.generator(key)
           timestep = restart(observation=self.__state_to_observation(state))
           return state, timestep
       
       
    def step(
        self, state: types_.State, action: chex.Numeric
    ) -> Tuple[types_.State, TimeStep[types_.Observation]]:
        """Run one timestep of the environment's dynamics.
        
        
        Args:
            state: 'State' object containing the dynamics of the environment.
            action: 'Array' containing the index of the next node to visit.
            
            
        Returns:
            state: the next state of the environment.
            timestep: the timestep to be observed.    
        """
        # Valid if node has not been visited and/or there is sufficient travel budget
        is_valid = ~state.visited_mask[action] & (state.length[action] < state.remaining_max_length)

        next_state = jax.lax.cond(
            is_valid,
            self._update_state,
            lambda *_: state,
            state,
            action,
        )
        
        reward = self.reward_fn(state, action, next_state, is_valid)
        observation = self._state_to_observation(next_state)
        
        # Terminate if all the travel budget is used up or the action is invalid
        is_done = (state.remaining_max_length == next_state.length[action]) | ~is_valid
        
        timestep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep
    
    
    def observation_spec(self) -> specs.Spec[types_.Observation]:
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
            self.num_nodes, dtype=jnp.int32, name="position"
        )
        trajectory = specs.BoundedArray(
            shape=(self.num_nodes, ),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_nodes,
            name="trajectory",
        )
        prizes = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="prizes",
        )
        length = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            minimum=0.0,
            maximum=2.0,
            dtype=float,
            name="length"
        )
        remaining_max_length = specs.BoundedArray(
            shape=(), minimum=0.0, maximum=2.0, dtype=float, name="remaining_max_length"
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_nodes + 1, ),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            types_.Observation,
            "ObservationSpec",
            coordinates=coordinates,
            position=position,
            trajectory=trajectory,
            prizes=prizes,
            length=length,
            remaining_max_length=remaining_max_length,
            action_mask=action_mask,
        )
        
        
    def action_spec(self) -> specs.DiscreteArray:
        """"Returns the action spec.
        
        Returns:
            action_spec: a 'specs.DiscreteArray' array.
        """
        
        return specs.DiscreteArray(self.num_nodes + 1, name="action")
    
    
    #def render(self)
    
    #def animate(self)
    
    #def close(self)
    
    
    def _update_state(self, state: types_.State, action: chex.Numeric) -> types_.State:
        """
        Updates the state of the environment.
        
        Args:
            state: State object containing the dynamics of the environment.
            action: int32, index of the next position to visit.
            
        Returns:
            state: State object corresponding to the new state of the environment.    
        """  
        
        return types_.State(
            coordinates=state.coordinates,
            position=state.position,
            visited_mask=state.visited_mask.at[action].set(True),
            trajectory=state.trajectory.at[state.num_visited].set(action),
            num_visited=state.num_visited + 1,
            prizes=state.prizes,
            length=state.prizes,
            remaining_max_length=state.remaining_max_length - state.length[action],
            key=state.key,
            
        )
        
     
    def _state_to_observation(self, state: types_.State) -> types_.Observation:
        """Converts a state to an observation.
        
        Args:
        state: State object containing the dynamics of the environment.
        
        
    Returns:
        observation: Observation object containing the observation of the environment.    
        """  
         
        # a node is reachable if it has not been visited already or if there is 
        # enough travel budget to cover the node's length to the depot
        action_mask = ~state.visited_mask & (state.length < state.remaining_max_length)
        
        return types_.Observation(
            coordinates=state.coordinates,
            position=state.position,
            trajectory=state.trajectory,
            prizes=state.prizes,
            length=state.length,
            remaining_max_length=state.remaining_max_length,
            action_mask=action_mask,
        )
        