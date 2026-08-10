from contextlib import contextmanager

import torch

from pretrain_src.pretrain_src.utils.training import (
    forward_backward_microbatch,
)


class _RecordingDdpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.no_sync_active = False
        self.no_sync_calls = 0
        self.forward_states = []
        self.backward_states = []

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        self.no_sync_active = True
        try:
            yield
        finally:
            self.no_sync_active = False

    def forward(self, batch, task, compute_loss):
        assert task == "mlm"
        assert compute_loss is True
        self.forward_states.append(self.no_sync_active)
        loss = self.weight * batch
        loss.register_hook(
            lambda gradient: (
                self.backward_states.append(self.no_sync_active)
                or gradient
            )
        )
        return loss


def test_pretrain_gradient_accumulation_only_syncs_final_microbatch():
    model = _RecordingDdpModel()
    batch = torch.ones(2)

    first_loss, first_units = forward_backward_microbatch(
        model,
        batch,
        "mlm",
        False,
        None,
        accumulation_steps=2,
        sync_gradients=False,
    )
    second_loss, second_units = forward_backward_microbatch(
        model,
        batch,
        "mlm",
        False,
        None,
        accumulation_steps=2,
        sync_gradients=True,
    )

    assert model.no_sync_calls == 1
    assert model.forward_states == [True, False]
    assert model.backward_states == [True, False]
    assert first_units == second_units == 2
    assert first_loss.item() == second_loss.item() == 0.5
    assert model.weight.grad.item() == 1.0
