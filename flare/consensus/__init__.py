"""
Flare Consensus Module

This module provides consensus mechanisms for distributed decision making
in federated learning networks.
"""

from .consensus import ConsensusMechanism
from .vrf_consensus import VRFConsensus

__all__ = [
    "ConsensusMechanism",
    "VRFConsensus",
]
