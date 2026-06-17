# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:46:11 2026

@author: alian
"""

# Test imports
import networkx as nx
from jax import numpy as jnp, random as jr, tree_util as jtu
import jaxtyping


import os
import pymdp
print(os.listdir(os.path.dirname(pymdp.__file__)))

from pymdp.envs import PymdpEnv
print("PymdpEnv imported successfully!")

# Hello World
print("Hello, World!")
print("networkx version:", nx.__version__)
print("JAX imported successfully!")
print("pymdp imported successfully!")
print(dir(pymdp))
