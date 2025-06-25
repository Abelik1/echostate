import torch
import pytest
from echostate import ESN

def generate_dummy_data(T, base_input_dim, output_dim):
    # simple linear relationship: y_t = sum(u_t) + noise
    inputs = torch.randn(T, base_input_dim)
    targets = torch.sum(inputs, dim=1, keepdim=True)[:]
    return inputs, targets

@pytest.fixture
def simple_batch():
    batch_size = 2
    T = 50
    base_input_dim = 3
    output_dim = 1
    input_list = []
    target_list = []
    for _ in range(batch_size):
        inp, tgt = generate_dummy_data(T, base_input_dim, output_dim)
        input_list.append(inp)
        # targets align after washout=0
        target_list.append(tgt)
    return input_list, target_list

@ pytest.mark.parametrize("feedback", [0, 1, 2])
def test_esn_fit_forward_predict(simple_batch, feedback):
    input_list, target_list = simple_batch
    T, base_input_dim = input_list[0].shape
    output_dim = target_list[0].shape[1]
    esn = ESN(base_input_dim=base_input_dim,
              reservoir_size=20,
              output_dim=output_dim,
              feedback=feedback,
              washout=5,
              batch_size=len(input_list),
              seed=42)
    # Training
    esn.fit(input_list, target_list)
    assert esn.W_out is not None
    # Forward on single sequence
    seq = input_list[0]
    out = esn.forward(seq)
    assert out.shape == (T - esn.washout, output_dim)
    # Predict on batch
    preds = esn.predict(input_list)
    assert isinstance(preds, list)
    assert preds[0].shape == (T - esn.washout, output_dim)