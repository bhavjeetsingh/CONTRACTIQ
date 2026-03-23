from utils.file_io import generate_session_id, validate_file_type
from utils.config_loader import load_config

def test_session_id_format():
    sid = generate_session_id("test")
    assert sid.startswith("test_")
    assert len(sid) > 10

def test_validate_file_type():
    assert validate_file_type("contract.pdf") == True
    assert validate_file_type("contract.exe") == False

def test_config_loads():
    config = load_config()
    assert "llm" in config
    assert "embedding_model" in config
