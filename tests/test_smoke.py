from src.config import ROOT, load_config, use_azure


def test_config_loads():
    cfg = load_config()
    assert cfg["audio"]["language"] == "en"
    assert "chest pain" in cfg["audio"]["critical_terms"]["chest_pain"]["patterns"]
    assert cfg["vision"]["frame_stride"] >= 1


def test_azure_off_by_default(monkeypatch):
    monkeypatch.delenv("USE_AZURE", raising=False)
    assert use_azure() is False


def test_structure_exists():
    for d in ["ingestion", "vision", "audio", "cloud", "fusion", "report"]:
        assert (ROOT / "src" / d / "__init__.py").exists()
