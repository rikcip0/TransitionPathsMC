
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def outdir():
    p = Path(__file__).resolve().parent / "_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    print(f"[ARTIFACTS DIR] {p.resolve()}")
    return p
