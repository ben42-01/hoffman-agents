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

__version__ = "2.0.0"

from .agent import ConsciousAgent, SimpleWorld, WorldState, StepOutput, ExperienceSpace
from .world import World, WorldBuilder, SelfWorld, CoinTossWorld, build_world_from_dataframe
from .network import AgentNetwork
from .combination import combine, trivial_agent, experience_space_distance
from .io import serialize, deserialize, clone, clone_agent, fingerprint, save_agent, load_agent, load_latest
from .core import (
    TraceBuffer,
    TraceEvent,
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    ExperienceLexicon,
    LexiconEntry,
    strange_loop_score,
    compute_self_reference_score,
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
    "StepOutput",
    "ExperienceSpace",
    "World",
    "WorldBuilder",
    "SelfWorld",
    "CoinTossWorld",
    "build_world_from_dataframe",
    "AgentNetwork",
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
    "MetaTrie",
    "SelfTokenState",
    "ExperienceLexicon",
    "LexiconEntry",
    "strange_loop_score",
    "compute_self_reference_score",
    "SharedMeaningTracker",
]
