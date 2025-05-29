from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ConsensusMechanism(ABC):
    """
    Abstract base class for consensus mechanisms in Flare.

    Consensus mechanisms handle decision making and validation
    in the federated learning network, ensuring agreement on
    model updates and network state.
    """

    def __init__(self):
        """Initialize the consensus mechanism."""
        pass

    @abstractmethod
    def select_committee(
        self, available_nodes: List[str], round_number: int, **kwargs
    ) -> List[str]:
        """
        Select committee members for validation.

        Args:
            available_nodes: List of available node IDs
            round_number: Current round number
            **kwargs: Additional context data

        Returns:
            List of selected committee member IDs
        """
        pass

    @abstractmethod
    def propose_decision(self, proposal_data: Dict[str, Any], proposer_id: str) -> str:
        """
        Propose a decision to the consensus mechanism.

        Args:
            proposal_data: Data about the proposal
            proposer_id: ID of the proposing node

        Returns:
            Proposal ID for tracking
        """
        pass

    @abstractmethod
    def vote(self, proposal_id: str, voter_id: str, vote: bool, **kwargs) -> bool:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: ID of the proposal
            voter_id: ID of the voting node
            vote: True for approve, False for reject
            **kwargs: Additional voting data

        Returns:
            True if vote was accepted, False otherwise
        """
        pass

    @abstractmethod
    def get_consensus_result(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the consensus result for a proposal.

        Args:
            proposal_id: ID of the proposal

        Returns:
            Consensus result or None if not yet decided
        """
        pass
