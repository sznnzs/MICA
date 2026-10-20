from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from typing import Any, Literal, Mapping


PROTOCOL_VERSION = "mica-interface"
SurfaceType = Literal["native", "cli", "mcp", "pure"]
ValueType = Literal["enum", "open_string", "object_ref", "boolean", "integer"]
DomainMode = Literal["exact", "examples", "open"]
_RESERVED_CONTAINER_PATTERN = re.compile(
    r"</?(?:action|command|tool_call)\s*>", flags=re.IGNORECASE
)


def contains_reserved_container_tag(value: Any) -> bool:
    return bool(_RESERVED_CONTAINER_PATTERN.search(str(value)))


@dataclass(frozen=True)
class NormalizationPolicy:
    strip: bool = True
    collapse_whitespace: bool = True
    case_sensitive: bool = False

    def apply(self, value: Any) -> str:
        text = str(value)
        if self.strip:
            text = text.strip()
        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text)
        return text if self.case_sensitive else text.casefold()


@dataclass(frozen=True)
class ArgumentSpec:
    argument_id: str
    value_type: ValueType
    required: bool = True
    domain_mode: DomainMode = "open"
    public_domain: tuple[str, ...] = ()
    domain_ref: str | None = None
    description: str = ""


@dataclass(frozen=True)
class SharedDomain:
    domain_id: str
    domain_mode: DomainMode
    values: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ActionSchema:
    schema_id: str
    operation_name: str
    arguments: tuple[ArgumentSpec, ...]
    native_template: str
    command_path: tuple[str, ...]
    tool_name: str
    availability_mode: Literal[
        "exact_instances", "exact_tuples", "schema_constrained", "backend_authoritative"
    ]
    upstream_template_id: str | None = None
    accepted_native_aliases: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ActionInstance:
    instance_id: str
    schema_id: str
    arguments: Mapping[str, Any]
    native_action_raw: str
    public: bool = True


@dataclass(frozen=True)
class AllowedTuple:
    schema_id: str
    arguments: Mapping[str, Any]
    native_action_raw: str | None = None
    evidence_id: str = ""
    evidence_object_ids: tuple[str, ...] = ()
    evidence_type_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationCapabilities:
    representability_basis: Literal["exact_catalog", "official_grammar", "hybrid"]
    availability_basis: Literal[
        "exact_catalog", "partial_evidence", "schema_only", "backend_authoritative"
    ]
    backend_rejection_is_authoritative: bool
    evidence_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActionSpaceIR:
    protocol_version: str
    contract_profile: str
    benchmark_id: str
    benchmark_version: str
    state_id: str
    source_kind: Literal["enumerated", "grammar", "hybrid"]
    schemas: tuple[ActionSchema, ...]
    instances: tuple[ActionInstance, ...]
    allowed_tuples: tuple[AllowedTuple, ...]
    validation_capabilities: ValidationCapabilities
    normalization: NormalizationPolicy
    public_fact_ids: tuple[str, ...]
    source_action_count: int
    prompt_catalog_mode: Literal[
        "full_exact",
        "compact_exact",
        "visible_schema_backend",
    ] = "full_exact"
    source_combination_count: int = 0
    upstream_source_combination_count: int = 0
    filtered_source_combination_count: int = 0
    represented_source_combination_count: int = 0
    combination_evidence_materialized: bool = True
    shared_domains: tuple[SharedDomain, ...] = ()
    object_referent_aliases: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    unmatched_source_actions: tuple[str, ...] = ()
    ambiguous_source_actions: tuple[str, ...] = ()
    unmatched_source_combinations: tuple[str, ...] = ()
    ambiguous_source_combinations: tuple[str, ...] = ()

    def schema(self, schema_id: str) -> ActionSchema | None:
        return next((item for item in self.schemas if item.schema_id == schema_id), None)

    @cached_property
    def normalized_allowed_tuple_index(
        self,
    ) -> Mapping[tuple[str, tuple[tuple[str, str], ...]], tuple[AllowedTuple, ...]]:
        """Index exact tuples without adding cache state to serialized records."""

        rows: dict[
            tuple[str, tuple[tuple[str, str], ...]], list[AllowedTuple]
        ] = {}
        for item in self.allowed_tuples:
            key = (
                item.schema_id,
                tuple(
                    sorted(
                        (name, self.normalization.apply(value))
                        for name, value in item.arguments.items()
                    )
                ),
            )
            rows.setdefault(key, []).append(item)
        return {key: tuple(values) for key, values in rows.items()}

    @cached_property
    def normalized_native_candidate_index(
        self,
    ) -> Mapping[str, tuple["ActionCandidate", ...]]:
        """Map every exact native spelling to its ID-grounded semantic tuple."""

        rows: dict[str, list[ActionCandidate]] = {}

        def add(candidate: ActionCandidate, spellings: tuple[str, ...]) -> None:
            semantic_key = (
                candidate.schema_id,
                tuple(
                    sorted(
                        (name, self.normalization.apply(value))
                        for name, value in candidate.arguments.items()
                    )
                ),
            )
            for spelling in spellings:
                key = self.normalization.apply(spelling)
                existing = rows.setdefault(key, [])
                existing_keys = {
                    (
                        item.schema_id,
                        tuple(
                            sorted(
                                (name, self.normalization.apply(value))
                                for name, value in item.arguments.items()
                            )
                        ),
                    )
                    for item in existing
                }
                if semantic_key not in existing_keys:
                    existing.append(candidate)

        for instance in self.instances:
            if not instance.public:
                continue
            schema = self.schema(instance.schema_id)
            if schema is None or schema.availability_mode != "exact_instances":
                continue
            arguments = dict(instance.arguments)
            add(
                ActionCandidate(
                    instance.schema_id,
                    arguments,
                    "native",
                    instance.native_action_raw,
                ),
                (
                    instance.native_action_raw,
                    schema.native_template.format(**arguments),
                    *(alias.format(**arguments) for alias in schema.accepted_native_aliases),
                ),
            )
        for item in self.allowed_tuples:
            schema = self.schema(item.schema_id)
            if schema is None or schema.availability_mode != "exact_tuples":
                continue
            arguments = dict(item.arguments)
            raw = str(item.native_action_raw or schema.native_template.format(**arguments))
            add(
                ActionCandidate(item.schema_id, arguments, "native", raw),
                (
                    raw,
                    schema.native_template.format(**arguments),
                    *(alias.format(**arguments) for alias in schema.accepted_native_aliases),
                ),
            )
        return {key: tuple(values) for key, values in rows.items()}

    def instances_for(self, schema_id: str, *, public_only: bool = False) -> tuple[ActionInstance, ...]:
        return tuple(
            item
            for item in self.instances
            if item.schema_id == schema_id and (item.public or not public_only)
        )

    def shared_domain(self, domain_id: str | None) -> SharedDomain | None:
        if domain_id is None:
            return None
        return next((item for item in self.shared_domains if item.domain_id == domain_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record_dict(self) -> dict[str, Any]:
        """Return a compact audit record with tuple counts and metadata.

        Official combination tuples are checked in memory; the compact record
        stores their count separately from the exact-catalog payload.
        """

        value = asdict(self)
        value["allowed_tuple_count"] = len(self.allowed_tuples)
        value["allowed_tuples_recorded"] = False
        value["combination_evidence_materialized"] = False
        value["allowed_tuples"] = []
        return value

    def to_exact_catalog_dict(self) -> dict[str, Any]:
        """Dictionary-encode exact tuples for compact, lossless audit replay."""

        catalogs: list[dict[str, Any]] = []
        for schema in self.schemas:
            items = [item for item in self.allowed_tuples if item.schema_id == schema.schema_id]
            if not items or schema.availability_mode != "exact_tuples":
                continue
            argument_order = [argument.argument_id for argument in schema.arguments]
            variants = (schema.native_template, *schema.accepted_native_aliases)
            tuple_values: list[list[Any]] = []
            variant_indices: list[int] = []
            evidence_object_ids: list[list[str]] = []
            evidence_type_ids: list[list[str]] = []
            evidence_ids: list[str] = []
            native_literal_overrides: list[dict[str, Any]] = []
            for item in items:
                arguments = dict(item.arguments)
                tuple_values.append([arguments[name] for name in argument_order])
                evidence_object_ids.append(list(item.evidence_object_ids))
                evidence_type_ids.append(list(item.evidence_type_ids))
                evidence_ids.append(str(item.evidence_id or ""))
                normalized_raw = self.normalization.apply(item.native_action_raw or "")
                matching = [
                    index
                    for index, template in enumerate(variants)
                    if self.normalization.apply(template.format(**arguments)) == normalized_raw
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"exact tuple does not select one native variant: {item.evidence_id}"
                    )
                variant_index = matching[0]
                variant_indices.append(variant_index)
                canonical_literal = variants[variant_index].format(**arguments)
                actual_literal = str(item.native_action_raw or "")
                if actual_literal != canonical_literal:
                    native_literal_overrides.append(
                        {
                            "tuple_index": len(tuple_values) - 1,
                            "native_action_raw": actual_literal,
                        }
                    )
            catalogs.append(
                {
                    "schema_id": schema.schema_id,
                    "argument_order": argument_order,
                    "argument_tuple_values": tuple_values,
                    "native_variant_indices": variant_indices,
                    "evidence_object_ids": evidence_object_ids,
                    "evidence_type_ids": evidence_type_ids,
                    "evidence_ids": evidence_ids,
                    "native_literal_overrides": native_literal_overrides,
                }
            )
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "state_id": self.state_id,
            "tuple_count": sum(
                len(item["argument_tuple_values"]) for item in catalogs
            ),
            "catalogs": catalogs,
        }

    def materialize_exact_catalog(self, value: Mapping[str, Any]) -> "ActionSpaceIR":
        """Restore exact tuples from a recorded dictionary-encoded sidecar."""

        if str(value.get("benchmark_id") or "") != self.benchmark_id:
            raise ValueError("exact catalog benchmark does not match contract")
        if str(value.get("benchmark_version") or "") != self.benchmark_version:
            raise ValueError("exact catalog benchmark version does not match contract")
        if str(value.get("state_id") or "") != self.state_id:
            raise ValueError("exact catalog state does not match contract")
        restored: list[AllowedTuple] = []
        for raw_catalog in value.get("catalogs") or ():
            catalog = dict(raw_catalog)
            schema_id = str(catalog.get("schema_id") or "")
            schema = self.schema(schema_id)
            if schema is None:
                raise ValueError(f"exact catalog references unknown schema: {schema_id}")
            argument_order = [str(item) for item in catalog.get("argument_order") or ()]
            expected_order = [argument.argument_id for argument in schema.arguments]
            if argument_order != expected_order:
                raise ValueError(f"exact catalog argument order differs: {schema_id}")
            rows = list(catalog.get("argument_tuple_values") or ())
            variants = list(catalog.get("native_variant_indices") or ())
            object_id_rows = list(catalog.get("evidence_object_ids") or ())
            type_id_rows = list(catalog.get("evidence_type_ids") or ())
            raw_evidence_ids = catalog.get("evidence_ids")
            evidence_ids = (
                [str(item) for item in raw_evidence_ids]
                if isinstance(raw_evidence_ids, list)
                else [""] * len(rows)
            )
            if not (
                len(rows)
                == len(variants)
                == len(object_id_rows)
                == len(type_id_rows)
                == len(evidence_ids)
            ):
                raise ValueError(
                    f"exact catalog tuple/variant/evidence lengths differ: {schema_id}"
                )
            native_templates = (schema.native_template, *schema.accepted_native_aliases)
            overrides: dict[int, str] = {}
            for raw_override in catalog.get("native_literal_overrides") or ():
                override = dict(raw_override)
                tuple_index = int(override.get("tuple_index", -1))
                literal = override.get("native_action_raw")
                if (
                    not 0 <= tuple_index < len(rows)
                    or tuple_index in overrides
                    or not isinstance(literal, str)
                    or not literal
                ):
                    raise ValueError(
                        f"exact catalog native literal override is invalid: {schema_id}"
                    )
                overrides[tuple_index] = literal
            for tuple_index, (
                row,
                raw_variant,
                raw_object_ids,
                raw_type_ids,
                evidence_id,
            ) in enumerate(
                zip(
                    rows,
                    variants,
                    object_id_rows,
                    type_id_rows,
                    evidence_ids,
                )
            ):
                values = list(row)
                if len(values) != len(argument_order):
                    raise ValueError(f"exact catalog tuple width differs: {schema_id}")
                variant_index = int(raw_variant)
                if not 0 <= variant_index < len(native_templates):
                    raise ValueError(f"exact catalog native variant is invalid: {schema_id}")
                arguments = dict(zip(argument_order, values))
                canonical_literal = native_templates[variant_index].format(**arguments)
                native_literal = overrides.get(tuple_index, canonical_literal)
                if self.normalization.apply(native_literal) != self.normalization.apply(
                    canonical_literal
                ):
                    raise ValueError(
                        f"exact catalog native literal changes semantics: {schema_id}"
                    )
                object_ids = tuple(str(item) for item in list(raw_object_ids))
                type_ids = tuple(str(item) for item in list(raw_type_ids))
                if (
                    object_ids
                    and len(object_ids) != len(argument_order)
                ) or (
                    type_ids and len(type_ids) != len(argument_order)
                ):
                    raise ValueError(
                        f"exact catalog evidence width differs: {schema_id}"
                    )
                restored.append(
                    AllowedTuple(
                        schema_id=schema_id,
                        arguments=arguments,
                        native_action_raw=native_literal,
                        evidence_id=evidence_id,
                        evidence_object_ids=object_ids,
                        evidence_type_ids=type_ids,
                    )
                )
        if int(value.get("tuple_count") or 0) != len(restored):
            raise ValueError("exact catalog tuple_count differs from restored rows")
        return replace(
            self,
            allowed_tuples=tuple(restored),
            combination_evidence_materialized=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionSpaceIR":
        """Reconstruct the canonical contract for independent validation."""

        schemas: list[ActionSchema] = []
        for raw_schema in value.get("schemas") or ():
            row = dict(raw_schema)
            row["arguments"] = tuple(
                ArgumentSpec(
                    **{
                        **dict(raw_argument),
                        "public_domain": tuple(
                            dict(raw_argument).get("public_domain") or ()
                        ),
                    }
                )
                for raw_argument in row.get("arguments") or ()
            )
            row["command_path"] = tuple(row.get("command_path") or ())
            row["accepted_native_aliases"] = tuple(
                row.get("accepted_native_aliases") or ()
            )
            schemas.append(ActionSchema(**row))
        instances = tuple(
            ActionInstance(
                **{
                    **dict(raw),
                    "arguments": dict(dict(raw).get("arguments") or {}),
                }
            )
            for raw in value.get("instances") or ()
        )
        allowed_tuples = tuple(
            AllowedTuple(
                schema_id=str(dict(raw).get("schema_id") or ""),
                arguments=dict(dict(raw).get("arguments") or {}),
                native_action_raw=(
                    str(dict(raw).get("native_action_raw"))
                    if dict(raw).get("native_action_raw") is not None
                    else None
                ),
                evidence_id=str(dict(raw).get("evidence_id") or ""),
                evidence_object_ids=tuple(
                    str(item)
                    for item in dict(raw).get("evidence_object_ids") or ()
                ),
                evidence_type_ids=tuple(
                    str(item)
                    for item in dict(raw).get("evidence_type_ids") or ()
                ),
            )
            for raw in value.get("allowed_tuples") or ()
        )
        capabilities = ValidationCapabilities(
            **{
                **dict(value.get("validation_capabilities") or {}),
                "evidence_source_ids": tuple(
                    dict(value.get("validation_capabilities") or {}).get(
                        "evidence_source_ids"
                    )
                    or ()
                ),
            }
        )
        normalization = NormalizationPolicy(**dict(value.get("normalization") or {}))
        shared_domains = tuple(
            SharedDomain(
                **{
                    **dict(raw),
                    "values": tuple(dict(raw).get("values") or ()),
                }
            )
            for raw in value.get("shared_domains") or ()
        )
        return cls(
            protocol_version=str(value.get("protocol_version") or ""),
            contract_profile=str(value.get("contract_profile") or "unprofiled"),
            benchmark_id=str(value.get("benchmark_id") or ""),
            benchmark_version=str(value.get("benchmark_version") or ""),
            state_id=str(value.get("state_id") or ""),
            source_kind=str(value.get("source_kind") or ""),  # type: ignore[arg-type]
            schemas=tuple(schemas),
            instances=instances,
            allowed_tuples=allowed_tuples,
            validation_capabilities=capabilities,
            normalization=normalization,
            public_fact_ids=tuple(value.get("public_fact_ids") or ()),
            source_action_count=int(value.get("source_action_count") or 0),
            prompt_catalog_mode=str(
                value.get("prompt_catalog_mode") or "full_exact"
            ),  # type: ignore[arg-type]
            source_combination_count=int(value.get("source_combination_count") or 0),
            upstream_source_combination_count=int(
                value.get("upstream_source_combination_count") or 0
            ),
            filtered_source_combination_count=int(
                value.get("filtered_source_combination_count") or 0
            ),
            represented_source_combination_count=int(
                value.get("represented_source_combination_count") or 0
            ),
            combination_evidence_materialized=bool(
                value.get("combination_evidence_materialized", True)
            ),
            shared_domains=shared_domains,
            object_referent_aliases={
                str(object_id): tuple(str(item) for item in values or ())
                for object_id, values in dict(
                    value.get("object_referent_aliases") or {}
                ).items()
            },
            unmatched_source_actions=tuple(value.get("unmatched_source_actions") or ()),
            ambiguous_source_actions=tuple(value.get("ambiguous_source_actions") or ()),
            unmatched_source_combinations=tuple(
                value.get("unmatched_source_combinations") or ()
            ),
            ambiguous_source_combinations=tuple(
                value.get("ambiguous_source_combinations") or ()
            ),
        )


@dataclass(frozen=True)
class ActionCandidate:
    schema_id: str
    arguments: Mapping[str, Any]
    surface: SurfaceType
    raw_proposal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisibilityCertificate:
    interface_type: SurfaceType
    public_fact_ids: tuple[str, ...]
    renderer_version: str = PROTOCOL_VERSION


@dataclass(frozen=True)
class SurfaceProjection:
    interface_type: SurfaceType
    semantic_catalog: str
    guidance: str
    response_contract: str
    visibility: VisibilityCertificate
    schema_count: int
    instance_count: int


@dataclass(frozen=True)
class FailureRecord:
    error_id: str
    first_failed_stage: Literal[
        "generation",
        "surface_decode",
        "schema_validate",
        "current_availability",
        "native_compile",
        "recovery_policy",
        "backend_step",
        "adapter_conformance",
        "infrastructure",
    ]
    violated_invariant: str
    obligation_owner: Literal[
        "policy_output", "interface_adapter", "benchmark_backend", "infrastructure", "ambiguous"
    ]
    raw_proposal: str
    parsed_candidate: Mapping[str, Any] | None
    transition_status: Literal["unchanged", "advanced", "unknown"]
    retryable: bool
    message: str
    allowed_values: tuple[str, ...] = ()
    public_correction_facts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterfaceOutcome:
    accepted: bool
    candidate: ActionCandidate | None
    native_action: str | None
    failure: FailureRecord | None


@dataclass(frozen=True)
class ConformanceReport:
    protocol_version: str
    benchmark_id: str
    state_id: str
    representability_basis: str
    availability_basis: str
    source_action_count: int
    source_combination_count: int
    upstream_source_combination_count: int
    filtered_source_combination_count: int
    represented_source_combination_count: int
    schema_count: int
    instance_count: int
    allowed_tuple_count: int
    surface_roundtrip_candidate_count: int
    unmatched_source_actions: tuple[str, ...] = ()
    ambiguous_source_actions: tuple[str, ...] = ()
    unmatched_source_combinations: tuple[str, ...] = ()
    ambiguous_source_combinations: tuple[str, ...] = ()
    normalization_collisions: tuple[str, ...] = ()
    native_roundtrip_failures: tuple[str, ...] = ()
    surface_roundtrip_failures: tuple[str, ...] = ()
    visibility_failures: tuple[str, ...] = ()
    negative_test_failures: tuple[str, ...] = ()
    status: Literal["pass", "fail"] = field(init=False)

    def __post_init__(self) -> None:
        failures = (
            self.unmatched_source_actions,
            self.ambiguous_source_actions,
            self.unmatched_source_combinations,
            self.ambiguous_source_combinations,
            self.normalization_collisions,
            self.native_roundtrip_failures,
            self.surface_roundtrip_failures,
            self.visibility_failures,
            self.negative_test_failures,
        )
        object.__setattr__(self, "status", "fail" if any(failures) else "pass")

    def require_pass(self) -> None:
        if self.status != "pass":
            raise AdapterConformanceError(
                f"interface contract failed conformance for {self.benchmark_id}/{self.state_id}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdapterConformanceError(RuntimeError):
    """The adapter cannot safely expose the current benchmark action source."""


class SurfaceDecodeError(ValueError):
    def __init__(
        self,
        error_id: str,
        message: str,
        *,
        allowed_values: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.error_id = error_id
        self.allowed_values = allowed_values


class CandidateValidationError(ValueError):
    def __init__(
        self,
        stage: Literal["schema_validate", "current_availability", "native_compile"],
        error_id: str,
        message: str,
        *,
        allowed_values: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_id = error_id
        self.allowed_values = allowed_values
