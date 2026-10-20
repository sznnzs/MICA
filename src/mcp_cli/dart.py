"""Validated typed semantic commitments for DART."""
from dataclasses import dataclass
import json
from .interfaces.ir import ActionCandidate
from .interfaces.engine import _validate_and_compile
from .interfaces.factorized import (COMMITMENT_PROTOCOL, canonical_json,
    contract_identity, typed_action)

@dataclass(frozen=True)
class Commitment:
    schema: str
    contract_id: str
    semantic_json: str
    candidate_json: str

    @classmethod
    def create(cls,candidate,contract):
        key=typed_action(candidate,contract)
        # Store canonical argument values for every present typed argument.
        args={k:v["value"] for k,v in key["arguments"].items() if v["present"]}
        canonical=ActionCandidate(candidate.schema_id,args,candidate.surface,"")
        _validate_and_compile(canonical,contract)
        return cls(COMMITMENT_PROTOCOL,contract_identity(contract),canonical_json(key),canonical_json({"schema_id":candidate.schema_id,"arguments":args}))

    def candidate(self,surface):
        value=json.loads(self.candidate_json)
        return ActionCandidate(value["schema_id"],value["arguments"],surface,"")
