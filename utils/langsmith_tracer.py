import os

from logger import GLOBAL_LOGGER as log


def setup_langsmith() -> bool:
    """Configure LangSmith tracing via environment variables.

    Tracing is enabled only when `LANGCHAIN_TRACING_V2` is true-ish
    and `LANGCHAIN_API_KEY` is present.
    """
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower()
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()

    enabled = tracing_flag in {"1", "true", "yes", "on"} and bool(api_key)
    if not enabled:
        log.info(
            "LangSmith tracing disabled",
            tracing_flag=tracing_flag,
            has_api_key=bool(api_key),
        )
        return False

    # LangChain reads these env vars at runtime for observability.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "CONTRACTIQ")

    log.info(
        "LangSmith tracing enabled",
        project=os.getenv("LANGCHAIN_PROJECT", "CONTRACTIQ"),
    )
    return True
