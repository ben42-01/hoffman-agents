"""conscious-agent — A computational implementation of Conscious Realism.

One import. One world. The agent does the rest.

Quick start:
    from conscious_agent import ConsciousAgent
    from conscious_agent.worlds import CoinTossWorld

    world = CoinTossWorld(n_coins=4)
    agent = ConsciousAgent(world=world, agent_id="my_agent")

    outputs = agent.run(n_steps=1000)
    print(f'"I" locked: {agent.experience.self_token.locked}')
"""

__version__ = "2.1.2"

from .agent import ConsciousAgent, SimpleWorld, WorldState, EnvironmentState, sequence_to_state_id, StepOutput, Prediction, ExperienceSpace
from .world import World, WorldBuilder, SelfWorld, CoinTossWorld, Normalizer, FeatureSpec, build_world_from_dataframe
from .network import AgentNetwork, Topology, InteractionCycle, Connection
from .combination import combine, trivial_agent, experience_space_distance, fuse
from .io import serialize, deserialize, clone, clone_agent, fingerprint, save_agent, load_agent, load_latest
from .core import (
    TraceBuffer,
    TraceEvent,
    ExperienceTrie,
    TrieNode,
    MetaTrie,
    MetaStateSnapshot,
    SelfTokenState,
    ExperienceLexicon,
    LexiconEntry,
    strange_loop_score,
    compute_self_reference_score,
    population_reference_score,
    population_loop_score,
    first_depth_n_generation,
    prune,
    trace_distance,
    merge_similar_paths,
    invent_token,
    is_invented_token,
)
from .meaning import SharedMeaningTracker

# Alias for CA_RUNTIME_API.md compatibility
import sys as _sys
_sys.modules['conscious_agent.worlds'] = _sys.modules['conscious_agent.world']
_sys.modules['conscious_agent.io'] = _sys.modules['conscious_agent.io']

__all__ = [
    "ConsciousAgent",
    "SimpleWorld",
    "WorldState",
    "EnvironmentState",
    "sequence_to_state_id",
    "StepOutput",
    "Prediction",
    "ExperienceSpace",
    "World",
    "WorldBuilder",
    "SelfWorld",
    "CoinTossWorld",
    "Normalizer",
    "FeatureSpec",
    "build_world_from_dataframe",
    "AgentNetwork",
    "Topology",
    "InteractionCycle",
    "Connection",
    "combine",
    "trivial_agent",
    "experience_space_distance",
    "serialize",
    "deserialize",
    "clone",
    "clone_agent",
    "fingerprint",
    "save_agent",
    "load_agent",
    "load_latest",
    "TraceBuffer",
    "TraceEvent",
    "ExperienceTrie",
    "TrieNode",
    "MetaTrie",
    "MetaStateSnapshot",
    "SelfTokenState",
    "ExperienceLexicon",
    "LexiconEntry",
    "strange_loop_score",
    "compute_self_reference_score",
    "population_reference_score",
    "population_loop_score",
    "first_depth_n_generation",
    "prune",
    "trace_distance",
    "merge_similar_paths",
    "invent_token",
    "is_invented_token",
    "SharedMeaningTracker",
]
