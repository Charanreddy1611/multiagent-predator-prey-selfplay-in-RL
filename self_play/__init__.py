"""
Self-play training strategies for multi-agent learning.
"""

from .alternating import AlternatingSelfPlay
from .population import PopulationSelfPlay
from .league import LeagueSelfPlay
from .reservoir import ReservoirSelfPlay

__all__ = ['AlternatingSelfPlay', 'PopulationSelfPlay', 'LeagueSelfPlay', 'ReservoirSelfPlay']

