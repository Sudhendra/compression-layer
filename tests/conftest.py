"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_verbose_text():
    """Sample verbose text for testing compression."""
    return """
    John Smith is a senior software engineer at Google who has been working
    on the search ranking team for the past three years. He previously worked
    at Microsoft for five years as a backend developer. His primary expertise
    is in distributed systems and machine learning infrastructure.
    """


@pytest.fixture
def sample_compressed_text():
    """Sample compressed text for testing."""
    return "John Smith | sr SWE @ Google | search ranking 3yr | prev: Microsoft 5yr backend | expertise: distributed systems, ML infra"


@pytest.fixture
def sample_verbose_code():
    """Sample verbose code for testing compression."""
    return '''
def calculate_total_price(items: list[dict], tax_rate: float = 0.08) -> float:
    """Calculate total price including tax."""
    subtotal = 0
    for item in items:
        subtotal += item['price'] * item['quantity']
    tax = subtotal * tax_rate
    return subtotal + tax
'''


@pytest.fixture
def sample_compressed_code():
    """Sample compressed code for testing."""
    return "fn:calculate_total_price(items, tax_rate=0.08)->float = sum(i.price*i.qty for i in items) |> λx: x*(1+tax_rate)"
