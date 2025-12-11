"""
Self-play training strategies for multi-agent learning.
"""

from .alternating import AlternatingSelfPlay
from .population import PopulationSelfPlay
from .league import LeagueSelfPlay

__all__ = ['AlternatingSelfPlay', 'PopulationSelfPlay', 'LeagueSelfPlay']

