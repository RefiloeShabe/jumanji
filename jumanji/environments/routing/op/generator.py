import abc
import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.op.types import State
from jumanji.environments.routing.op.constants import DEPOT_IDX

class Generator(abc.ABC):
    """
    Defines the abstract 'Generator' base class. A 'Generator' is responsible
    for generating a problem instance when the environment is reset
    """

    def __init__(self, num_nodes: int, max_length: int):
        """Abstract class implementing the attributes 'num_nodes' and 'max_length'.
        
        Args:
            num_nodes (int): the number of nodes in the problem instance.
            max_length (int): the maximum length of the tour
        """
        
        self.num_nodes = num_nodes
        self.max_length = max_length
        
      
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.
        Args:
            key: jax random key in case stochasticity is used in the instance generation
            process
            
        Returns:
            An 'OP' environment state.
        """    
        
    def generate_base_attributes(self, key: chex.PRNGKey):  
        key, coordinates_key, length_key = jax.random.split(key, num=3)
        
        # Randomly sample the coordinates of the nodes
        coordinates = jax.random.uniform(
            coordinates_key, (self.num_nodes + 1, 2), minval=0, maxval=1
        )  
        
        # The initial position is set at the depot
        position = jnp.array(DEPOT_IDX, jnp.int32)
        
        # Initially, the agent has ony visited the depot
        visited_mask = jnp.zeros(self.num_nodes + 1, dtype=bool).at[DEPOT_IDX].set(True)
        trajectory = jnp.full(self.num_nodes, DEPOT_IDX, jnp.int32)
        
        # The number of visited nodes
        num_visited = jnp.array(1, jnp.int32)
        
        # Randomly sample the length of nodes from the depot
        length = jax.random.uniform(
            length_key, (self.num_nodes + 1, ), minval=0, maxval=2)
        
        # Set depot length to 0 --> length between depot and depot is 0
        length = length.at[DEPOT_IDX].set(0)
        
        # The remaining travel budget
        remaining_max_length = jnp.array(self.max_length, float)  
        
        return coordinates, position, visited_mask, trajectory, num_visited, length, remaining_max_length, key
    
    def generate_prizes(self, key: chex.PRNGKey):
        _, _, _, _, _, length, _, _ = self.generate_base_attributes(key)
        
        # Compute prizes -> prizes are propotional to the distance to the depot, set prize to 0 at depot 
        prizes = 0.01 + (0.99 * length / jnp.max(length)).at[DEPOT_IDX].set(0)
        
        return prizes      
        
        
class ConstantGenerator(Generator):
    """Instance generator that generates a constant instance of the orienteering problem.
    Given the number of nodes and maximum route length, the generator works as follows: the 
    coordinates of the nodes (including the depot) are randomly sampled from a  uniform 
    distribution of a unit square and the prizes are set constant such that every node has the 
    same prize. The prize at the depot is 0. The length between the depot and the nodes  is 
    randomly sampled from the interval (0, max_length).
    """
    
    def __call__(self, key: chex.PRNGKey) -> State:
        coordinates, position, visited_mask, trajectory, num_visited, length, remaining_max_length, key = self.generate_base_attributes(key)
        prizes = jnp.ones(self.num_nodes + 1).at[DEPOT_IDX].set(0)
        
        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            prizes=prizes,
            length=length,
            remaining_max_length=remaining_max_length,
            key=key,
        )
        
        return state
    

class UniformGenerator(Generator):
    """Instance generator that generates a random uniform instance of the orienteering problem.
    Given the number of nodes and maximum route length, the generator works as follows: the 
    coordinates of the nodes (including the depot) and the prizes are randomly sampled from a 
    uniform distribution of the unit square. The prize at the depot is 0. The length between the 
    depot and the nodes is randomly sampled from the interval (0, max_length).
    """
         
    def __call__(self, key: chex.PRNGKey) -> State:
        coordinates, position, visited_mask, trajectory, num_visited, length, remaining_max_length, key = self.generate_base_attributes(key)
        key, prize_key = jax.random.split(key)
        prizes = jax.random.uniform(
            prize_key, (self.num_nodes + 1, ), minval=0, maxval=1).at[DEPOT_IDX].set(0)
        
        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            prizes=prizes,
            length=length,
            remaining_max_length=remaining_max_length,
            key=key,
        )
        
        return state  
          
    
class ProportionalGenerator(Generator):
    """Instance generator that generates an instance of the orienteering problem where every node
    has a prize that is proportional to the distance to the depot. Given the number of nodes and 
    maximum route length, the generator works as follows: the  coordinates of the nodes (including the depot) 
    are randomly sampled from a uniform distribution of the unit square. The length between the depot and the 
    nodes is randomly sampled from the interval (0, max_length). The prizes of the nodes are calculated as a 
    discretized proportion of the length between the depot and the nodes.
    """
    
    def __call__(self, key: chex.PRNGKey) -> State:
        key, prize_key = jax.random.split(key)
        coordinates, position, visited_mask, trajectory, num_visited, length, remaining_max_length, key = self.generate_base_attributes(key)
        prizes = self.generate_prizes(prize_key)
        
        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            prizes=prizes,
            length=length,
            remaining_max_length=remaining_max_length,
            key=key,
        )
        
        return state             