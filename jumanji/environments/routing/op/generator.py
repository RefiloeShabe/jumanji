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
      
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.
        Args:
            key: jax random key in case stochasticity is used in the instance generation
            process
            
        Returns:
            An 'OP' environment state.
        """     
        key, coordinates_key, length_key, prize_key = jax.random.split(key, num=4)
        
        # Randomly sample the coordinates of the nodes
        coordinates = jax.random.uniform(
            coordinates_key, (self.num_nodes + 1, 2), minval=0, maxval=1
        )  
        
        # The initial position is set at the depot
        position = jnp.array(DEPOT_IDX, jnp.int32)
        
        # Initially, the agent has ony visited the depot
        visited_mask = jnp.zeros(self.num_nodes + 1, dtype=bool).at[DEPOT_IDX].set(False)
        trajectory = jnp.full(self.num_nodes, DEPOT_IDX, jnp.int32)
        
        # The number of visited nodes
        num_visited = jnp.array(1, jnp.int32)
        
        # Randomly sample the length of nodes from the depot
        length = jax.random.uniform(
            length_key, (self.num_nodes + 1, ), minval=0, maxval=2)
        
        # Set depot length to 0 --> length between depot and depot is 0
        length = length.at[DEPOT_IDX].set(0)
        
        prizes = self._generate_prizes(prize_key, length)
        
        # The remaining travel budget
        remaining_max_length = jnp.array(self.max_length, float)  
        
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
    
    @abc.abstractmethod
    def _generate_prizes(self, key: chex.PRNGKey, length):
        raise NotImplementedError      
  
        
class ConstantGenerator(Generator):
    """Instance generator that generates a constant instance of the orienteering problem.
    Given the number of nodes and maximum route length, the generator works as follows: the 
    coordinates of the nodes (including the depot) are randomly sampled from a  uniform 
    distribution of a unit square and the prizes are set constant such that every node has the 
    same prize. The prize at the depot is 0. The length between the depot and the nodes  is 
    randomly sampled from the interval (0, max_length).
    """
    
    def _generate_prizes(self, _key, _length):

        # All nodes have the same costant prize,  set depot prize to 0
        prizes = jnp.ones(self.num_nodes + 1).at[DEPOT_IDX].set(0)

        return prizes        
  
        
class UniformGenerator(Generator):
    """Instance generator that generates a random uniform instance of the orienteering problem.
    Given the number of nodes and maximum route length, the generator works as follows: the 
    coordinates of the nodes (including the depot) and the prizes are randomly sampled from a 
    uniform distribution of the unit square. The prize at the depot is 0. The length between the 
    depot and the nodes is randomly sampled from the interval (0, max_length).
    """

    def _generate_prizes(self, key, _length):

        # All nodes have the same costant prize,  set depot prize to 0
        prizes = jax.random.uniform(
            key, (self.num_nodes + 1, ), minval=0, maxval=1).at[DEPOT_IDX].set(0)

        return prizes 
    
        
class ProportionalGenerator(Generator):
    """Instance generator that generates an instance of the orienteering problem where every node
    has a prize that is proportional to the distance to the depot. Given the number of nodes and 
    maximum route length, the generator works as follows: the  coordinates of the nodes (including the depot) 
    are randomly sampled from a uniform distribution of the unit square. The length between the depot and the 
    nodes is randomly sampled from the interval (0, max_length). The prizes of the nodes are calculated as a 
    discretized proportion of the length between the depot and the nodes.
    """

    def _generate_prizes(self, _key, length):

        # All nodes have the same costant prize,  set depot prize to 0
        prizes = 0.01 + (0.99 * length / jnp.max(length))
        prizes = prizes.at[DEPOT_IDX].set(0)

        return prizes 