import hashlib
import json
import random
import time
from typing import Any, Dict, List, Optional

from .consensus import ConsensusMechanism


class VRFConsensus(ConsensusMechanism):
    """
    Verifiable Random Function (VRF) Consensus Mechanism for Flare.

    This consensus mechanism uses VRF to:
    1. Select committee members for model validation
    2. Ensure fair and verifiable committee selection
    3. Validate aggregated models through committee voting
    4. Prevent single point of failure in model validation

    Algorithm:
    - Use VRF to generate verifiable random committee selection
    - Committee members validate aggregated models
    - Consensus achieved through majority vote
    - Results are cryptographically verifiable
    """

    def __init__(
        self,
        committee_size: int = 5,
        min_committee_threshold: float = 0.6,
        vrf_seed: Optional[str] = None,
    ):
        """
        Initialize VRF consensus mechanism.

        Args:
            committee_size: Number of committee members to select
            min_committee_threshold: Minimum fraction of committee agreement needed
            vrf_seed: Optional seed for VRF generation (for simulation reproducibility)
        """
        super().__init__()
        self.committee_size = committee_size
        self.min_committee_threshold = min_committee_threshold
        self.vrf_seed = vrf_seed or str(int(time.time()))

        # State tracking
        self.current_committee: List[str] = []
        self.committee_votes: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []

        print(
            f"VRFConsensus initialized with committee_size={committee_size}, "
            f"threshold={min_committee_threshold}"
        )

    def select_committee(
        self,
        available_nodes: List[str],
        round_number: int,
        block_data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Select committee members using VRF for given round.

        Args:
            available_nodes: List of available node IDs
            round_number: Current round number
            block_data: Optional blockchain context data

        Returns:
            List of selected committee member IDs
        """
        if len(available_nodes) < self.committee_size:
            print(
                f"VRFConsensus: Warning - Only {len(available_nodes)} nodes available, "
                f"selecting all as committee"
            )
            self.current_committee = available_nodes.copy()
            return self.current_committee

        # Generate VRF input for this round
        vrf_input = self._generate_vrf_input(round_number, block_data)

        # Use VRF to select committee members
        selected_committee = self._vrf_select_committee(available_nodes, vrf_input)

        self.current_committee = selected_committee

        print(
            f"VRFConsensus: Selected committee for round {round_number}: {selected_committee}"
        )
        return selected_committee

    def propose_decision(self, proposal_data: Dict[str, Any], proposer_id: str) -> str:
        """
        Propose a decision (e.g., model validation) to the committee.

        Args:
            proposal_data: Data about the proposal (model hash, metadata, etc.)
            proposer_id: ID of the node making the proposal

        Returns:
            Proposal ID for tracking
        """
        proposal_id = self._generate_proposal_id(proposal_data, proposer_id)

        proposal = {
            "proposal_id": proposal_id,
            "proposer_id": proposer_id,
            "proposal_data": proposal_data,
            "timestamp": int(time.time()),
            "committee": self.current_committee.copy(),
            "votes": {},
            "status": "pending",
        }

        # Initialize vote tracking for this proposal
        self.committee_votes[proposal_id] = proposal

        print(f"VRFConsensus: Proposal {proposal_id} submitted by {proposer_id}")
        print(f"VRFConsensus: Committee {self.current_committee} will vote on proposal")

        return proposal_id

    def vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote: bool,
        vote_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: ID of the proposal to vote on
            voter_id: ID of the voting committee member
            vote: True for approve, False for reject
            vote_data: Optional additional voting data (validation results, etc.)

        Returns:
            True if vote was accepted, False otherwise
        """
        if proposal_id not in self.committee_votes:
            print(f"VRFConsensus: Proposal {proposal_id} not found")
            return False

        proposal = self.committee_votes[proposal_id]

        # Verify voter is in committee
        if voter_id not in proposal["committee"]:
            print(f"VRFConsensus: {voter_id} not authorized to vote on {proposal_id}")
            return False

        # Record vote
        proposal["votes"][voter_id] = {
            "vote": vote,
            "timestamp": int(time.time()),
            "vote_data": vote_data or {},
        }

        print(f"VRFConsensus: {voter_id} voted {vote} on proposal {proposal_id}")

        # Check if consensus is reached
        self._check_consensus(proposal_id)

        return True

    def get_consensus_result(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the consensus result for a proposal.

        Args:
            proposal_id: ID of the proposal

        Returns:
            Consensus result or None if not yet decided
        """
        if proposal_id not in self.committee_votes:
            return None

        proposal = self.committee_votes[proposal_id]

        if proposal["status"] == "pending":
            return None

        return {
            "proposal_id": proposal_id,
            "result": proposal["status"],
            "votes": proposal["votes"],
            "committee": proposal["committee"],
            "timestamp": proposal.get("decision_timestamp", 0),
        }

    def _generate_vrf_input(
        self, round_number: int, block_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate VRF input for committee selection.

        Args:
            round_number: Current round number
            block_data: Optional blockchain context

        Returns:
            VRF input string
        """
        # Combine round number, seed, and optional block data
        input_components = [str(round_number), self.vrf_seed]

        if block_data:
            # Include relevant blockchain data for stronger randomness
            if "previous_block_hash" in block_data:
                input_components.append(block_data["previous_block_hash"])
            if "timestamp" in block_data:
                input_components.append(str(block_data["timestamp"]))

        vrf_input = "|".join(input_components)
        return vrf_input

    def _vrf_select_committee(
        self, available_nodes: List[str], vrf_input: str
    ) -> List[str]:
        """
        Use VRF to select committee members.

        Args:
            available_nodes: List of available node IDs
            vrf_input: VRF input string

        Returns:
            Selected committee member IDs
        """
        # Generate deterministic random seed from VRF input
        vrf_hash = hashlib.sha256(vrf_input.encode()).hexdigest()
        vrf_seed_int = int(vrf_hash[:16], 16)  # Use first 16 hex chars as seed

        # Use seeded random selection for reproducible committee selection
        random.seed(vrf_seed_int)

        # Select committee members without replacement
        selected = random.sample(available_nodes, self.committee_size)

        # Reset random seed to avoid affecting other randomness
        random.seed()

        return selected

    def _generate_proposal_id(
        self, proposal_data: Dict[str, Any], proposer_id: str
    ) -> str:
        """
        Generate unique proposal ID.

        Args:
            proposal_data: Proposal data
            proposer_id: Proposer node ID

        Returns:
            Unique proposal ID
        """
        # Create deterministic ID from proposal content
        content = (
            json.dumps(proposal_data, sort_keys=True)
            + proposer_id
            + str(int(time.time()))
        )
        proposal_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"proposal_{proposal_hash[:16]}"

    def _check_consensus(self, proposal_id: str) -> None:
        """
        Check if consensus has been reached for a proposal.

        Args:
            proposal_id: ID of the proposal to check
        """
        proposal = self.committee_votes[proposal_id]

        if proposal["status"] != "pending":
            return

        votes = proposal["votes"]
        committee = proposal["committee"]

        # Check if all committee members have voted
        if len(votes) == len(committee):
            # All votes cast - determine result
            self._finalize_consensus(proposal_id)
        else:
            # Check if enough votes to reach threshold
            approve_votes = sum(1 for vote_info in votes.values() if vote_info["vote"])
            total_committee = len(committee)

            # Can we reach consensus with remaining votes?
            max_possible_approvals = approve_votes + (total_committee - len(votes))
            min_required_approvals = (
                int(total_committee * self.min_committee_threshold) + 1
            )

            if approve_votes >= min_required_approvals:
                # Consensus reached - approved
                proposal["status"] = "approved"
                proposal["decision_timestamp"] = int(time.time())
                print(
                    f"VRFConsensus: Proposal {proposal_id} APPROVED (early consensus)"
                )
            elif max_possible_approvals < min_required_approvals:
                # Cannot reach approval threshold - rejected
                proposal["status"] = "rejected"
                proposal["decision_timestamp"] = int(time.time())
                print(
                    f"VRFConsensus: Proposal {proposal_id} REJECTED (early consensus)"
                )

    def _finalize_consensus(self, proposal_id: str) -> None:
        """
        Finalize consensus result when all votes are cast.

        Args:
            proposal_id: ID of the proposal to finalize
        """
        proposal = self.committee_votes[proposal_id]
        votes = proposal["votes"]
        committee = proposal["committee"]

        approve_votes = sum(1 for vote_info in votes.values() if vote_info["vote"])
        total_votes = len(votes)
        approval_rate = approve_votes / total_votes

        if approval_rate >= self.min_committee_threshold:
            proposal["status"] = "approved"
        else:
            proposal["status"] = "rejected"

        proposal["decision_timestamp"] = int(time.time())
        proposal["approval_rate"] = approval_rate

        # Add to consensus history
        consensus_result = {
            "proposal_id": proposal_id,
            "result": proposal["status"],
            "approval_rate": approval_rate,
            "committee_size": len(committee),
            "votes_cast": total_votes,
            "timestamp": proposal["decision_timestamp"],
        }

        self.consensus_history.append(consensus_result)

        print(
            f"VRFConsensus: Proposal {proposal_id} {proposal['status'].upper()} "
            f"(approval rate: {approval_rate:.1%})"
        )

    def get_consensus_stats(self) -> Dict[str, Any]:
        """
        Get statistics about consensus performance.

        Returns:
            Dictionary with consensus statistics
        """
        if not self.consensus_history:
            return {
                "total_proposals": 0,
                "approved": 0,
                "rejected": 0,
                "average_approval_rate": 0.0,
            }

        total = len(self.consensus_history)
        approved = sum(
            1 for result in self.consensus_history if result["result"] == "approved"
        )
        rejected = total - approved
        avg_approval_rate = (
            sum(result["approval_rate"] for result in self.consensus_history) / total
        )

        return {
            "total_proposals": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total if total > 0 else 0.0,
            "average_approval_rate": avg_approval_rate,
            "current_committee": self.current_committee,
            "committee_size": self.committee_size,
        }
