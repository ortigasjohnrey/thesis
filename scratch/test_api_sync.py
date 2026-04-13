import torch
import torch.nn as nn
from api import CNN_BiLSTM, SelfAttention

def test_api_arch():
    print("Testing API Architecture Sync...")
    model = CNN_BiLSTM(input_dim=7, hidden_dim=128, filters=64)
    x = torch.randn(1, 30, 7)
    out = model(x)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 1), "Output shape should be (1, 1)"
    print("Architecture Test Passed!")

if __name__ == "__main__":
    try:
        test_api_arch()
    except Exception as e:
        print(f"Test Failed: {e}")
