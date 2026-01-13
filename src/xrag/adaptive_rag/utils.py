from pathlib import Path
from typing import Any, Optional


def extract_token_usage(response_obj: Any) -> Optional[int]:
    """
    Extract token usage (input + output tokens) from LLM response objects.

    Args:
        response_obj: The response object from LLM or query engine

    Returns:
        Total token count (input + output) if available, None otherwise
    """
    if response_obj is None:
        return None

    total_tokens = None

    # Try to get from raw response (common for OpenAI-style responses)
    if hasattr(response_obj, "raw") and response_obj.raw is not None:
        raw = response_obj.raw
        if hasattr(raw, "usage"):
            usage = raw.usage
            if hasattr(usage, "total_tokens"):
                return usage.total_tokens
            elif hasattr(usage, "prompt_tokens") and hasattr(
                usage, "completion_tokens"
            ):
                return usage.prompt_tokens + usage.completion_tokens
        # Handle dict-style raw response
        if isinstance(raw, dict) and "usage" in raw:
            usage = raw["usage"]
            if isinstance(usage, dict):
                if "total_tokens" in usage:
                    return usage["total_tokens"]
                elif "prompt_tokens" in usage and "completion_tokens" in usage:
                    return usage["prompt_tokens"] + usage["completion_tokens"]

    # Try additional_kwargs (used by some LLM providers)
    if hasattr(response_obj, "additional_kwargs"):
        kwargs = response_obj.additional_kwargs
        if isinstance(kwargs, dict):
            if "usage" in kwargs:
                usage = kwargs["usage"]
                if isinstance(usage, dict):
                    if "total_tokens" in usage:
                        return usage["total_tokens"]
                    elif "prompt_tokens" in usage and "completion_tokens" in usage:
                        return usage["prompt_tokens"] + usage["completion_tokens"]
            # Direct token counts in additional_kwargs
            if "total_tokens" in kwargs:
                return kwargs["total_tokens"]
            elif "prompt_tokens" in kwargs and "completion_tokens" in kwargs:
                return kwargs["prompt_tokens"] + kwargs["completion_tokens"]

    # Try metadata (used by Response objects from query engines)
    if hasattr(response_obj, "metadata") and response_obj.metadata is not None:
        metadata = response_obj.metadata
        if isinstance(metadata, dict):
            # Look for token info in metadata
            for key in ["token_usage", "usage", "tokens"]:
                if key in metadata:
                    usage = metadata[key]
                    if isinstance(usage, int):
                        return usage
                    if isinstance(usage, dict):
                        if "total_tokens" in usage:
                            return usage["total_tokens"]
                        elif "prompt_tokens" in usage and "completion_tokens" in usage:
                            return usage["prompt_tokens"] + usage["completion_tokens"]

    # Try response_metadata (used by some response types)
    if hasattr(response_obj, "response_metadata"):
        resp_meta = response_obj.response_metadata
        if isinstance(resp_meta, dict):
            if "token_usage" in resp_meta:
                usage = resp_meta["token_usage"]
                if isinstance(usage, dict):
                    if "total_tokens" in usage:
                        return usage["total_tokens"]
                    elif "prompt_tokens" in usage and "completion_tokens" in usage:
                        return usage["prompt_tokens"] + usage["completion_tokens"]

    if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
        for node in response_obj.source_nodes:
            if hasattr(node, "metadata") and isinstance(node.metadata, dict):
                if "token_usage" in node.metadata:
                    usage = node.metadata["token_usage"]
                    if isinstance(usage, dict):
                        if "total_tokens" in usage:
                            return usage["total_tokens"]
                        elif "prompt_tokens" in usage and "completion_tokens" in usage:
                            return usage["prompt_tokens"] + usage["completion_tokens"]

    if hasattr(response_obj, "response") and response_obj.response is not None:
        underlying_tokens = extract_token_usage(response_obj.response)
        if underlying_tokens is not None:
            return underlying_tokens

    return total_tokens


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the adaptive_rag prompts directory.

    Args:
        prompt_name (str): Name of the prompt file (without .txt extension)

    Returns:
        The prompt template content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompts_dir = Path(__file__).parent.parent / "prompts" / "adaptive_rag"
    prompt_path = prompts_dir / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
