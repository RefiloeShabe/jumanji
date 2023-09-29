from typing import TYPE_CHECKING, NamedTuple

import chex
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass
    
      
@dataclass
class State:
    """
    coordinates: array of 2D coordinates of all nodes (including the depot) 
    position: index of the current node
    visited_mask: binary mask (False/True <--> unvisited/visited)
    trajectory: array of the node indices defining the route (set to depot 
    index if not filled yet)
    num_visited: number of total nodes visited
    prizes: array with the prizes associated with each node
    lengths: array with the length between the depot and each node
    remaining_length: current tour length budget 
    key: random key used for auto-reset
    """
    
    coordinates: chex.Array  # (num_nodes, 2)
    position: chex.Numeric  # ()
    visited_mask: chex.Array  # (num_nodes, )
    trajectory: chex.Array   # (num_nodes, )
    num_visited: chex.Numeric  # ()
    prizes: chex.Array  # (num_nodes, )
    length: chex.Array  # (num_nodes, )
    remaining_max_length: chex.Array  # ( )
    key:  chex.PRNGKey  # (2, )  
    
    
class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates of all nodes (including the depot)
    position: index of the current node
    trajectory: array of the node indices defining the route (set to depot 
    index if not filled yet)
    prizes: array with the prizes associated with each node
    lengths: array of the length between the depot and each node 
    remaining_length: current tour length budget 
    action_mask: binary mask (False/True <--> illegal/legal)
    """
    
    coordinates: chex.Array  # (num_nodes, 2)
    position: chex.Numeric  # ()
    trajectory: chex.Array   # (num_nodes, )
    prizes: chex.Array  # (num_nodes, )
    length: chex.Array  # (num_nodes, )
    remaining_max_length: chex.Array  #( )
    action_mask: chex.Array  # (num_nodes, )