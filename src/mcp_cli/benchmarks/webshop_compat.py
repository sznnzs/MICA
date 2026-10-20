from __future__ import annotations

import json
import re
import sys
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any


RESTRICTED_PROFILE_ID = "restricted_text"
PROFILE_IDS = frozenset({RESTRICTED_PROFILE_ID})
PROFILE_ID = RESTRICTED_PROFILE_ID
RETRIEVAL_BACKEND_ID = "lexical_token_overlap"
REWARD_NLP_BACKEND_ID = "all_tokens_as_nouns"


def install_compatibility_backend(
    webshop_root: Path,
    environment_profile: str = PROFILE_ID,
) -> None:
    """Install the explicitly configured dependency-light WebShop backend."""

    if environment_profile not in PROFILE_IDS:
        raise ValueError(
            f"unsupported WebShop environment_profile={environment_profile!r}"
        )

    lucene = types.ModuleType("pyserini.search.lucene")
    CompatibilitySearcher.webshop_root = webshop_root
    lucene.LuceneSearcher = CompatibilitySearcher
    search = types.ModuleType("pyserini.search")
    search.lucene = lucene
    pyserini = types.ModuleType("pyserini")
    pyserini.search = search
    sys.modules["pyserini"] = pyserini
    sys.modules["pyserini.search"] = search
    sys.modules["pyserini.search.lucene"] = lucene

    spacy = types.ModuleType("spacy")
    spacy.load = lambda _name: _SimpleNLP()  # type: ignore[attr-defined]
    sys.modules["spacy"] = spacy

    try:
        import transformers.utils as utils
        import transformers.utils.import_utils as import_utils

        original = import_utils._is_package_available

        def available(package_name: str, *args: object, **kwargs: object):
            if package_name == "flash_attn":
                return False
            return original(package_name, *args, **kwargs)

        import_utils._is_package_available = available
        import_utils.is_flash_attn_2_available = lambda: False
        utils.is_flash_attn_2_available = lambda: False
    except Exception:
        pass

    _install_reward_schema_compatibility()


def install_rich_observation_mode(environment_class=None) -> None:
    """Expose the upstream tagged-text renderer as the text_rich mode.

    The adapter uses ``convert_html_to_text(..., simple=False)`` to retain
    the renderer's text and clicked-option markers.
    """
    if environment_class is None:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        environment_class = WebAgentTextEnv
    if environment_class.__dict__.get("_mica_rich_observation", False):
        return
    original = environment_class.observation

    def observation(environment):
        if environment.observation_mode == "text_rich":
            return environment.convert_html_to_text(environment.state["html"], simple=False)
        return original.fget(environment)

    environment_class.observation = property(observation, doc=original.__doc__)
    environment_class._mica_rich_observation = True


def _goal_option_values(goal_options: Any) -> list[Any]:
    """Unwrap the mapping entries passed by the bundled reward evaluator."""

    if isinstance(goal_options, Mapping):
        return list(goal_options.values())
    if isinstance(goal_options, (str, bytes)):
        return [goal_options]
    if not isinstance(goal_options, Iterable):
        return [goal_options]
    values: list[Any] = []
    for item in goal_options:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            values.append(item[1])
        else:
            values.append(item)
    return values


def _install_reward_schema_compatibility() -> None:
    from web_agent_site.engine import goal as goal_module

    current = goal_module.get_option_reward
    if bool(getattr(current, "_mica_option_schema_compat", False)):
        return

    @wraps(current)
    def compatible_get_option_reward(
        purchased_options: Any,
        goal_options: Any,
    ) -> Any:
        return current(purchased_options, _goal_option_values(goal_options))

    compatible_get_option_reward._mica_option_schema_compat = True  # type: ignore[attr-defined]
    compatible_get_option_reward._mica_original = current  # type: ignore[attr-defined]
    goal_module.get_option_reward = compatible_get_option_reward


class _SimpleToken:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos_ = "NOUN"


class _SimpleNLP:
    def __call__(self, text: str) -> list[_SimpleToken]:
        return [_SimpleToken(item) for item in _tokens(text)]


@dataclass
class _Hit:
    docid: str
    score: float


class _Document:
    def __init__(self, asin: str) -> None:
        self.asin = asin

    def raw(self) -> str:
        return json.dumps({"id": self.asin})


class CompatibilitySearcher:
    webshop_root: Path | None = None
    _cache: dict[str, tuple[dict[str, frozenset[str]], dict[str, str]]] = {}

    def __init__(self, index_path: str) -> None:
        self.index_path = str(index_path)
        self.search_tokens, self.search_text = self._load(self.index_path)

    @classmethod
    def _load(cls, index_path: str) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
        if cls.webshop_root is None:
            raise RuntimeError("WebShop compatibility search root is not configured")
        cache_key = str(cls.webshop_root.resolve()) + ":" + index_path
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        data_dir = cls.webshop_root / "data"
        index_name = Path(index_path).name
        source_name = "items_shuffle.json" if index_name == "indexes" else "items_shuffle_1000.json"
        products = json.loads((data_dir / source_name).read_text(encoding="utf-8"))
        token_sets: dict[str, frozenset[str]] = {}
        texts: dict[str, str] = {}
        for product in products:
            asin = str(product.get("asin") or "").upper()
            if not asin or asin == "NAN" or len(asin) > 10:
                continue
            fields = []
            for key in (
                "name",
                "brand",
                "full_description",
                "product_category",
                "category",
                "query",
                "small_description",
            ):
                value = product.get(key)
                if isinstance(value, list):
                    fields.extend(str(item) for item in value)
                elif isinstance(value, dict):
                    fields.extend(str(item) for item in value.values())
                elif value is not None:
                    fields.append(str(value))
            tokens = _tokens(" ".join(fields))
            token_sets[asin] = frozenset(tokens)
            texts[asin] = " ".join(tokens)
        cls._cache[cache_key] = (token_sets, texts)
        return cls._cache[cache_key]

    def search(self, keywords: str, k: int = 50) -> list[_Hit]:
        tokens = _tokens(keywords)
        if not tokens:
            return []
        token_set = set(tokens)
        phrase = " ".join(tokens)
        hits = []
        for asin, product_tokens in self.search_tokens.items():
            score = len(token_set & product_tokens)
            if phrase and phrase in self.search_text.get(asin, ""):
                score += 2
            if score:
                hits.append(_Hit(asin, float(score)))
        hits.sort(key=lambda item: (-item.score, item.docid))
        return hits[:k]

    def doc(self, docid: str) -> _Document:
        return _Document(str(docid).upper())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())
