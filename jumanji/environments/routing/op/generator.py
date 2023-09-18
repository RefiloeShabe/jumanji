import abc
import chex
import jax
import jax.numpy as jnp

import types_, constants

class Generator(abc.ABC):
    """
    Defines the abstract 'Generator' base class. A 'Generator' is responsible
    for generating a problem instance when the environment is reset
    """

    def __init__(self, num_nodes: int, max_length: int):
        """Abstract class implementing the attribute 'num_nodes'.
        
        Args:
            num_nodes (int): the number of nodes in the problem instance.
            max_length (int): the maximum length of the tour
        """
        
        self.num_nodes = num_nodes
        self.max_length = max_length
        
      
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> types_.State:
        """Call method responsible for generating a new state.
        Args:
            key: jax random key in case stochasticity is used in the instance generation process
            
        Returns:
            An 'OP' environment state.
        """    
        
        
class UniformGenerator(Generator):
    """Instance generator that generates a random uniform instance of the orienteering problem.
    Given the number of nodes and maximum route length, the generator works as follows: the 
    coordinates of the nodes (including the depot) and the prizes are randomly sampled from a 
    uniform distribution of the unit square. The prize at the depot is 0. The length between the 
    depot and the nodes is randomly sampled from the interval (0, max_length).
    """
    
    def __init__(self, num_nodes: int, max_length: int):
        """Instantiates a 'UniformGenerator'.
        
        Args:
            num_nodes (int): the number of nodes in the problem instance.
            max_length (int): the maximum length of the tour
        """        
        super().__init__(num_nodes, max_length)
        
        
    def __call__(self, key: chex.PRNGKey) -> types_.State:
        key, coordinates_key, prizes_key, length_key = jax.random.split(key, num=4)
        
        # Randomly sample the coordinates of the nodes
        coordinates = jax.random.uniform(
            coordinates_key, (self.num_nodes + 1, 2), minval=0, maxval=1
        )  
        
        # The initial position is set at the depot
        position = jnp.array(constants.DEPOT_IDX, jnp.int32)
        
        # Initially, the agent has ony visited the depot
        visited_mask = jnp.zeros(self.num_nodes + 1, dtype=bool).at[constants.DEPOT_IDX].set(True)
        trajectory = jnp.full(self.num_nodes, constants.DEPOT_IDX, jnp.int32)
        
        # The number of visited nodes
        num_visited = jnp.array(1, jnp.int32)
        
        # Randomly sample the prizes 
        prizes = jax.random.uniform(
            prizes_key, (self.num_nodes + 1, ), minval=0, maxval=1)
        
        # Randomly sample the length of nodes from the depot
        length = jax.random.uniform(
            length_key, (self.num_nodes + 1, ), minval=0, maxval=2)
        
        # Set depot prize and lenght to 0 --> length between depot and depot is 0
        prizes = prizes.at[constants.DEPOT_IDX].set(0)
        length = length.at[constants.DEPOT_IDX].set(0)
        
        # The remaining travel budget
        remaining_max_length = jnp.array(self.max_length, float)
        
        state = types_.State(
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