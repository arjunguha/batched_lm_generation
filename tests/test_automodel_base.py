import pytest
from batched_lm_generation.automodel_base import AutoModelGenerator
import torch

@pytest.fixture
def automodel_generator():
    return AutoModelGenerator(model_name="openai-community/gpt2", include_prompt="", model_kwargs={})

def test_model_initialization(automodel_generator):
    assert automodel_generator.model_name == "openai-community/gpt2", "Model name should be 'openai-community/gpt2'"

def test_gpu_to_cpu_fallback(caplog):
    with caplog.at_level(logging.WARNING):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            assert "Warning: CUDA not available. Switching to CPU mode." in caplog.text
