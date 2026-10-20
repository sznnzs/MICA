from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .utils import estimate_tokens


CONSERVATIVE_UTF8_MODE = "conservative_utf8_bytes_per_3"
HF_CHAT_TEMPLATE_EXACT_MODE = "hf_chat_template_exact"
INPUT_TOKEN_COUNTING_MODES = frozenset(
    {CONSERVATIVE_UTF8_MODE, HF_CHAT_TEMPLATE_EXACT_MODE}
)
_TOKENIZER_CACHE: dict[Path, tuple[Any, threading.RLock]] = {}
_TOKENIZER_CACHE_LOCK = threading.Lock()


class InputTokenCounter(Protocol):
    mode: str

    def count_text(self, text: str) -> int:
        pass

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        pass


class ConservativeUtf8TokenCounter:
    """Tokenizer-independent fail-safe used when the provider tokenizer is unknown."""

    mode = CONSERVATIVE_UTF8_MODE

    def count_text(self, text: str) -> int:
        return estimate_tokens(text)

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        # Estimate message-content tokens. The configured input safety margin
        # accounts for roles and chat-template control tokens.
        return estimate_tokens(
            "".join(str(dict(message).get("content") or "") for message in messages)
        )


class HuggingFaceChatTemplateTokenCounter:
    """Exact local counter using the same tokenizer/chat template as vLLM."""

    mode = HF_CHAT_TEMPLATE_EXACT_MODE

    def __init__(
        self,
        *,
        tokenizer_path: str | Path,
        chat_template_kwargs: Mapping[str, Any],
    ) -> None:
        source = Path(tokenizer_path).expanduser().resolve(strict=True)
        try:
            # Use the direct implementation import for concurrent initialization.
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "a usable transformers AutoTokenizer is required for "
                f"hf_chat_template_exact: {exc}"
            ) from exc
        self.tokenizer_path = source
        self.chat_template_kwargs = dict(chat_template_kwargs)
        with _TOKENIZER_CACHE_LOCK:
            cached = _TOKENIZER_CACHE.get(source)
            if cached is None:
                cached = (
                    AutoTokenizer.from_pretrained(
                        source,
                        trust_remote_code=True,
                        local_files_only=True,
                    ),
                    threading.RLock(),
                )
                _TOKENIZER_CACHE[source] = cached
        self._tokenizer, self._lock = cached

    def count_text(self, text: str) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        with self._lock:
            return len(self._tokenizer.encode(raw, add_special_tokens=False))

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        normalized = [dict(message) for message in messages]
        with self._lock:
            token_ids = self._tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        return len(token_ids)


def input_token_counting_identity(
    model_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = model_config.get("input_token_counting")
    if not isinstance(raw, Mapping):
        return None
    value = dict(raw)
    mode = str(value.get("mode") or CONSERVATIVE_UTF8_MODE)
    result: dict[str, Any] = {"mode": mode}
    if mode == HF_CHAT_TEMPLATE_EXACT_MODE:
        result["tokenizer_id"] = str(value.get("tokenizer_id") or "")
        result["chat_template_kwargs"] = dict(
            dict(model_config.get("extra_body") or {}).get(
                "chat_template_kwargs"
            )
            or {}
        )
    return result


def build_input_token_counter(
    model_config: Mapping[str, Any],
) -> InputTokenCounter:
    raw = model_config.get("input_token_counting")
    value = dict(raw) if isinstance(raw, Mapping) else {}
    mode = str(value.get("mode") or CONSERVATIVE_UTF8_MODE)
    if mode == CONSERVATIVE_UTF8_MODE:
        return ConservativeUtf8TokenCounter()
    if mode != HF_CHAT_TEMPLATE_EXACT_MODE:
        raise ValueError(f"unsupported input token counting mode: {mode}")
    tokenizer_path = str(value.get("tokenizer_path") or "").strip()
    if not tokenizer_path:
        raise ValueError(
            "hf_chat_template_exact requires input_token_counting.tokenizer_path"
        )
    chat_template_kwargs = dict(
        dict(model_config.get("extra_body") or {}).get("chat_template_kwargs")
        or {}
    )
    return HuggingFaceChatTemplateTokenCounter(
        tokenizer_path=tokenizer_path,
        chat_template_kwargs=chat_template_kwargs,
    )
