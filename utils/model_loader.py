import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from utils.langsmith_tracer import setup_langsmith
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class ApiKeyManager:
    PROVIDER_KEYS = {
        "groq": "GROQ_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    def __init__(self, provider: str | None = None):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Collect all known key env vars
        all_env_keys = set(self.PROVIDER_KEYS.values())
        for key in all_env_keys:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Only require the key for the active provider
        active_key = self.PROVIDER_KEYS.get(provider or os.getenv("LLM_PROVIDER", "groq"), "GROQ_API_KEY")
        if not self.api_keys.get(active_key):
            log.error("Missing required API key for active provider", key=active_key, provider=provider)
            raise DocumentPortalException("Missing API key: " + active_key, sys)

        log.info("API key loaded", provider=provider, key=active_key)


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


_embeddings_cache = None

class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        setup_langsmith()

        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        provider = os.getenv("LLM_PROVIDER", "groq").strip()
        self.api_key_mgr = ApiKeyManager(provider=provider)

    def load_embeddings(self):
        global _embeddings_cache
        if _embeddings_cache is not None:
            log.info("Returning cached embedding model")
            return _embeddings_cache

        # Check if Google key is available to use high-performance, low-memory API embeddings (primary for cloud)
        google_api_key = None
        try:
            google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY")
        except Exception:
            pass

        if google_api_key:
            try:
                model_name = self.config.get("embedding_model", {}).get("model_name", "models/embedding-001")
                log.info("Loading high-performance Google Gemini embedding model (primary)", model=model_name)
                _embeddings_cache = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=google_api_key
                )
                return _embeddings_cache
            except Exception as google_error:
                log.warning("Google embedding initialization failed, falling back to HuggingFace", error=str(google_error))

        # Fallback to local HuggingFace (heavy, requires PyTorch & substantial RAM)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            log.info("Loading local HuggingFace BGE embedding model (fallback)")
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
            )
            return _embeddings_cache
        except Exception as hf_error:
            log.error("Both Google and HuggingFace embedding models failed to load", hf_error=str(hf_error))
            raise DocumentPortalException("Failed to load any embedding model", sys)

    def load_llm(self, provider_key: str | None = None):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = (provider_key or os.getenv("LLM_PROVIDER", "groq")).strip()

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
