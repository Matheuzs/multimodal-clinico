from src.config import ROOT, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["audio"]["language"] == "en"
    assert "chest pain" in cfg["audio"]["critical_terms"]["chest_pain"]["patterns"]
    assert cfg["vision"]["frame_stride"] >= 1



def test_structure_exists():
    for d in ["ingestion", "vision", "audio", "fusion", "report"]:
        assert (ROOT / "src" / d / "__init__.py").exists()
