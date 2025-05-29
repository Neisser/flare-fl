from typing import Any, List

from flare.core.base_classes import FlareConfig

from .consensus import ConsensusMechanism


class MockPoAConsensus(ConsensusMechanism):
    """
    Simulates a Proof-of-Authority-like consensus where a central entity (the orchestrator
    in this simulation) makes the decision, and it's logged.
    """
    def __init__(self, config: 'FlareConfig'):
        super().__init__(config)
        print("MockPoAConsensus initialized.")

    def validate_contribution(self, contribution: Any, **kwargs) -> bool:
        # In simulated PoA, the authority (orchestrator) implicitly validates by accepting.
        print(f"MockPoAConsensus: Validating contribution (assuming valid): {str(contribution)[:100]}...")
        # Siempre válido en este mock
        return True

    def reach_agreement(self, proposals: List[Any], **kwargs) -> Any:
        # In simulated PoA, if there are multiple proposals for a global model,
        # the "authority" (orchestrator) might simply choose the first one,
        # or the one it received (which is the one it already calculated).
        print(f"MockPoAConsensus: Reaching agreement on {len(proposals)} proposals.")
        if not proposals:
            print("MockPoAConsensus: No proposals to agree on.")
            return None
        # The "authority" (Orchestrator) has already done the aggregation, so the proposal is the result.
        agreed_result = proposals[0]
        print(f"MockPoAConsensus: Agreement reached (authority decision): {str(agreed_result)[:100]}")

        # Optional: simulate logging to blockchain
        # bc_connector = self.config.get('blockchain_connector')
        # if bc_connector:
        #     bc_connector.submit_transaction({
        #         "action": "log_agreement",
        #         "agreed_result_hash": hash(str(agreed_result)) # Simplistic hash
        #     })
        return agreed_result
