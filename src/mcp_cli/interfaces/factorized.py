"""Factorized Φ / R / Ψ prompts around the shared interface engine.

Rendering is deterministic and independent of model or environment execution.
The factorial prompt is distinct from the joint baseline prompt.
"""
from dataclasses import asdict, dataclass, replace
import json
from ..configuration import content_hash
from ..schemas import StepResult
from .ir import ActionCandidate, FailureRecord, SurfaceDecodeError
from .engine import InterfaceEngine, _validate_and_compile
from .feedback import visible_failure_payload
from .surfaces import (render_surface, encode_candidate,
                       exact_tuple_values_are_policy_visible, _flag, _load_strict_json)

PROTOCOL="mica-factorized"
COMMITMENT_PROTOCOL="typed-commitment"
HISTORY_PROTOCOL="public-semantic-ledger"
HISTORY_TREATMENTS={"fixed_r_strict", "source_attributed_lossless"}
CONDITIONS={f"C{i+1}":levels for i,levels in enumerate(
    ( (p,r,s) for p in ("mcp","cli") for r in ("mcp","cli") for s in ("mcp","cli") ))}


def _style(style):
    if style not in {"mcp","cli"}:raise ValueError("MCP or CLI block style required")


def canonical_json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def contract_identity(contract):
    # Hash the contract's state locator, full IR, and exact verifier catalog.
    return content_hash(asdict(contract))


def _baseline_public_catalog(contract):
    """Read the model-visible catalog from the shared surface renderer.

    Both compact and expanded catalog encodings are supported. Each record
    is checked against its expected schema.
    """
    lines=render_surface(contract,"mcp").semantic_catalog.split("\n")
    dictionaries=[json.loads(line[2:]) for line in lines if line.startswith("D=")]
    if len(dictionaries)>1:raise ValueError("ambiguous baseline value dictionary")
    dictionary=dictionaries[0] if dictionaries else None
    rows=[json.loads(line[2:]) for line in lines if line.startswith("- ")]
    feedback_prefix="Feedback schema after rejection: "
    feedback=[json.loads(line[len(feedback_prefix):]) for line in lines if line.startswith(feedback_prefix)]
    portable=[line for line in lines if line.startswith("Portable string rule: ")]
    if len(feedback)!=1 or len(portable)!=1:raise ValueError("baseline public protocol declarations missing or ambiguous")
    return dictionary,rows,feedback[0],portable[0]


def semantic_facts(contract):
    """Project the shared public catalog into semantic facts.

    Schema fields, argument domains, visible tuples, and published tuple counts
    retain their model-visible values.
    """
    dictionary,rows,_,_=_baseline_public_catalog(contract)
    schemas=[];domains=[]
    argument_keys=("id","type","required","domain_mode","domain_ref","public_domain")
    for row in rows:
        if "domain" in row:
            if set(row)!={"domain"} or len(row["domain"])!=3:raise ValueError("baseline public domain shape changed")
            domain_id,mode,value_ids=row["domain"]
            domains.append({"id":domain_id,"mode":mode,"values":[dictionary[i] for i in value_ids]})
            continue
        if "shared_domain_id" in row:
            if set(row)!={"shared_domain_id","domain_mode","values"}:raise ValueError("baseline public domain shape changed")
            domains.append({"id":row["shared_domain_id"],"mode":row["domain_mode"],"values":row["values"]})
            continue
        if "args" in row:
            if dictionary is None:raise ValueError("compact baseline catalog lacks its public dictionary")
            if set(row)!={"id","op","mode","args","order","tuples"}:raise ValueError("baseline public schema shape changed")
            if any(len(values)!=len(argument_keys) for values in row["args"]):raise ValueError("baseline public argument shape changed")
            if any(len(values)!=len(row["order"]) for values in row["tuples"]):raise ValueError("baseline public tuple shape changed")
            arguments=[dict(zip(argument_keys,values)) for values in row["args"]]
            for argument in arguments:argument["public_domain"]=[dictionary[i] for i in argument["public_domain"]]
            types={argument["id"]:argument["type"] for argument in arguments}
            tuples=[{name:dictionary[value] if type(value) is int and types[name] not in {"integer","boolean"} else value
                     for name,value in zip(row["order"],values)} for values in row["tuples"]]
            schema_id,operation,mode=row["id"],row["op"],row["mode"]
        elif "schema_id" in row:
            required={"schema_id","operation","availability_mode","arguments","current_argument_tuple_order","current_argument_tuple_values"}
            optional={"current_argument_tuple_count","tuple_values_presented","validation"}
            if not required<=set(row) or set(row)-required-optional:raise ValueError("baseline public schema shape changed")
            arguments=row["arguments"]
            if any(set(argument)!=set(argument_keys) for argument in arguments):raise ValueError("baseline public argument shape changed")
            if any(len(values)!=len(row["current_argument_tuple_order"]) for values in row["current_argument_tuple_values"]):raise ValueError("baseline public tuple shape changed")
            tuples=[dict(zip(row["current_argument_tuple_order"],values)) for values in row["current_argument_tuple_values"]]
            schema_id,operation,mode=row["schema_id"],row["operation"],row["availability_mode"]
        else:raise ValueError("unrecognized baseline public catalog record")
        schemas.append({"schema_id":schema_id,"operation":operation,"arguments":arguments,
                        "availability_mode":mode,"tuple_values_presented":row.get("tuple_values_presented",True),
                        "current_argument_tuple_count":row.get("current_argument_tuple_count",len(tuples)),
                        "current_argument_tuples":tuples,"validation":row.get("validation",mode)})
    return {"schemas":schemas,"shared_domains":domains}


def public_feedback_contract(contract):
    """Return the public rejection fields and correction allowance verbatim."""
    return _baseline_public_catalog(contract)[2]


def render_feedback_contract(contract,style):
    _style(style);facts=public_feedback_contract(contract)
    if style=="mcp":return canonical_json({"public_rejection_contract":facts})
    return "Public rejection contract:\n"+"\n".join(k+" = "+canonical_json(v) for k,v in facts.items())


def render_phi(contract,style):
    _style(style);facts=semantic_facts(contract)
    if style=="mcp":return canonical_json({"capabilities":facts})
    # Render the same semantic facts as deterministic, parseable CLI documentation.
    return "\n".join(["Semantic capabilities (not invocation syntax):",
                      *[k+" = "+canonical_json(v) for k,v in facts.items()]])


def phi_facts(text,style):
    _style(style)
    if style=="mcp":return json.loads(text)["capabilities"]
    return {k:json.loads(v) for k,v in (line.split(" = ",1) for line in text.split("\n")[1:])}


def realization_bindings(contract,style):
    _style(style)
    return [{"schema_id":s.schema_id,
             **({"tool_name":s.tool_name,"argument_keys":{a.argument_id:a.argument_id for a in s.arguments}} if style=="mcp" else
                {"command_path":list(s.command_path),"flags":{a.argument_id:_flag(a.argument_id) for a in s.arguments}})} for s in contract.schemas]


def render_r(contract,style):
    _style(style)
    rule=render_surface(contract,style).response_contract
    # Reuse the shared public string-portability rule.
    portable=_baseline_public_catalog(contract)[3]
    return rule+"\n"+portable+"\n"+canonical_json({"bindings":realization_bindings(contract,style)})


def projection(contract,condition_id):
    phi,rho,psi=CONDITIONS[condition_id]
    original=render_surface(contract,rho)
    catalog=render_phi(contract,phi)
    return replace(original,semantic_catalog=catalog,guidance="Φ\n"+catalog+"\nR\n"+render_r(contract,rho)+"\nΨ\n"+render_feedback_contract(contract,psi),
                   visibility=replace(original.visibility,renderer_version=PROTOCOL+":"+condition_id))


@dataclass(frozen=True)
class PublicResult:
    """Typed envelope for the complete policy-visible result.

    Public failure fields are copied verbatim through the visibility projection.
    """
    observation: str | None = None
    admission: bool | None = None
    execution_status: str = "not_called"
    error_id: str | None = None
    failure_stage: str | None = None
    retryable: bool | None = None
    transition_status: str = "unchanged"
    first_failed_stage: str | None = None
    violated_invariant: str | None = None
    message: str | None = None
    allowed_values: tuple[str, ...] | None = None
    correction: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.execution_status not in {"not_called","returned","unknown"}:raise ValueError("invalid public execution status")
        if self.transition_status not in {"unchanged","advanced","unknown"}:raise ValueError("invalid transition status")
        if self.observation is not None and not isinstance(self.observation,str):raise ValueError("public observation must be text")
        if self.admission is not None and type(self.admission) is not bool:raise ValueError("admission must be boolean or unknown")
        if self.retryable is not None and type(self.retryable) is not bool:raise ValueError("retryable must be boolean or unknown")
        if any(v is not None and not isinstance(v,str) for v in (
            self.error_id,self.failure_stage,self.first_failed_stage,self.violated_invariant,self.message
        )):raise ValueError("public failure fields must be text")
        if self.first_failed_stage is not None and self.first_failed_stage!=self.failure_stage:
            raise ValueError("public first_failed_stage must equal candidate failure_stage")
        for name in ("allowed_values","correction"):
            value=getattr(self,name)
            if value is not None:
                if not isinstance(value,(tuple,list)) or any(not isinstance(v,str) for v in value):
                    raise ValueError(name+" must be a sequence of public strings or unknown")
                # Freeze a caller-owned list before retaining it in PublicHistory.
                object.__setattr__(self,name,tuple(value))

    @classmethod
    def from_failure(cls,failure,*,observation=None,admission=False,execution_status="not_called"):
        if not isinstance(failure,FailureRecord):raise TypeError("explicit FailureRecord required")
        payload=visible_failure_payload(failure)
        return cls(observation=observation,admission=admission,execution_status=execution_status,
                   failure_stage=payload["first_failed_stage"],**payload)

    def public_facts(self):
        facts=asdict(self)
        for name in ("allowed_values","correction"):
            if facts[name] is not None:facts[name]=list(facts[name])
        return facts

    @property
    def has_surface_failure(self):
        return any(value is not None for value in (
            self.first_failed_stage,self.violated_invariant,self.message,self.allowed_values,self.correction
        ))

    @classmethod
    def from_step(cls,step):
        # Project the policy-visible observation and registered transition status.
        if not isinstance(step,StepResult):raise TypeError("explicit StepResult required")
        return cls(observation=str(step.policy_observation or step.observation),
                   execution_status="returned",transition_status=step.online_feedback.get("transition_status","unknown"))


def render_psi(result,style):
    _style(style)
    if not isinstance(result,PublicResult):raise TypeError("explicit PublicResult required; arbitrary raw dict forbidden")
    facts=result.public_facts()
    if style=="mcp":return canonical_json({"public_result":facts})
    return "Public result:\n"+"\n".join(k+" = "+canonical_json(v) for k,v in facts.items())


def psi_facts(text,style):
    _style(style)
    if style=="mcp":return json.loads(text)["public_result"]
    # splitlines() also splits legal Unicode line separators inside JSON strings.
    return {k:json.loads(v) for k,v in (line.split(" = ",1) for line in text.split("\n")[1:])}


def typed_action(candidate,contract,*,admitted=True):
    """Build a complete semantic key retaining each argument's type and value."""
    native=_validate_and_compile(candidate,contract) if admitted else None
    schema=contract.schema(candidate.schema_id)
    values=dict(candidate.arguments)
    if admitted:
        # Use a unique compiler-equivalent public binding when available;
        # otherwise retain the supplied typed values.
        public=[i for i in contract.instances if i.public and i.schema_id==schema.schema_id and i.native_action_raw==native]
        if len(public)==1:values=dict(public[0].arguments)
        else:
            public=[i for i in contract.allowed_tuples if i.schema_id==schema.schema_id and i.native_action_raw==native]
            if len(public)==1 and exact_tuple_values_are_policy_visible(contract,schema):values=dict(public[0].arguments)
    specs={a.argument_id:a for a in schema.arguments} if schema else {}
    arguments={}
    for name in sorted(set(specs)|set(values)):
        spec=specs.get(name);present=name in values;value=values.get(name)
        if isinstance(value,str) and admitted:
            value=value.strip()
            if spec and spec.domain_mode=="exact":
                matches=[v for v in spec.public_domain if contract.normalization.apply(v)==contract.normalization.apply(value)]
                if len(matches)==1:value=matches[0]
        arguments[name]={"declared_type":spec.value_type if spec else None,"present":present,
                         "observed_type":type(value).__name__ if present else None,"value":value if present else None}
    return {"schema_id":candidate.schema_id,"operation":schema.operation_name if schema else None,"arguments":arguments}


def decode_selection(raw,contract):
    value=_load_strict_json(raw,label="DART semantic selection")
    if not isinstance(value,dict) or set(value)!={"schema_id","arguments"} or not isinstance(value["schema_id"],str) or not isinstance(value["arguments"],dict):
        raise SurfaceDecodeError("selection.invalid_shape","selection requires schema_id and all typed arguments")
    return ActionCandidate(value["schema_id"],value["arguments"],"mcp",raw)


@dataclass(frozen=True)
class PublicEvent:
    event_index: int
    contract_id: str
    intent_status: str
    proposal_json: str | None
    result: PublicResult
    raw_output_ref: str | None
    source_rho: str


class PublicHistory:
    def __init__(self):self._events=[]
    @property
    def events(self):return tuple(self._events)

    def append(self,raw,contract,condition_id,result,*,raw_output_ref=None):
        if not isinstance(result,PublicResult):raise TypeError("public result envelope required")
        outcome=InterfaceEngine().process(raw,projection=projection(contract,condition_id),contract=contract)
        known=outcome.candidate is not None
        action=typed_action(outcome.candidate,contract,admitted=outcome.accepted) if known else None
        event=PublicEvent(len(self._events),contract_identity(contract),"KNOWN" if outcome.accepted else "KNOWN_INVALID" if known else "UNKNOWN",
                          canonical_json(action) if action is not None else None,result,raw_output_ref,CONDITIONS[condition_id][1])
        self._events.append(event);return event

    def ledger(self):
        """Return a JSON-safe ledger of public semantic events.

        Contract IDs identify recorded contracts; each entry stores an immutable
        semantic proposal and its public result.
        """
        return {"protocol":HISTORY_PROTOCOL,"events":[{
            "event_index":e.event_index,"contract_id":e.contract_id,
            "intent_status":e.intent_status,"proposal":json.loads(e.proposal_json) if e.proposal_json else None,
            "public_result":e.result.public_facts(),"source_rho":e.source_rho,
        } for e in self._events]}

    def render(self,contracts,*,rho,psi,history_treatment="fixed_r_strict"):
        _style(rho);_style(psi);rows=[]
        if history_treatment not in HISTORY_TREATMENTS:raise ValueError("explicit registered history treatment required")
        for e in self._events:
            contract=contracts[e.contract_id]
            if contract_identity(contract)!=e.contract_id:raise ValueError("history contract identity mismatch")
            if e.result.has_surface_failure and rho!=e.source_rho and history_treatment=="fixed_r_strict":
                raise ValueError("cross-R surface-specific failure history is unsupported: preserve all public fields; register a lossless surface-aware history treatment first")
            if e.intent_status=="UNKNOWN":expression="UNKNOWN (undecodable intent; canonical control history)"
            elif e.intent_status=="KNOWN_INVALID":expression="KNOWN_INVALID semantic proposal: "+e.proposal_json
            else:
                value=json.loads(e.proposal_json)
                candidate=ActionCandidate(value["schema_id"],{k:v["value"] for k,v in value["arguments"].items() if v["present"]},rho,"")
                expression=encode_candidate(candidate,contract,rho)
            provenance=""
            if history_treatment=="source_attributed_lossless":
                # Attribute verbatim error text to its recorded source surface.
                provenance=("\nRecorded public result from source R="+e.source_rho+
                            "; quoted source syntax is retained, not translated.\n")
            rows.append(f"Event {e.event_index}\n{expression}"+provenance+("" if provenance else "\n")+render_psi(e.result,psi))
        return "\n\n".join(rows)
