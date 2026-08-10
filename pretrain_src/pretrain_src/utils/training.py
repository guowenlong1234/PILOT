from contextlib import nullcontext

import torch.cuda.amp as amp


def forward_backward_microbatch(
    model,
    batch,
    task,
    fp16,
    grad_scaler,
    accumulation_steps,
    sync_gradients,
):
    """Run one microbatch, suppressing DDP sync until the final one."""
    sync_context = nullcontext() if sync_gradients else model.no_sync()
    with sync_context:
        if fp16:
            with amp.autocast():
                loss = model(batch, task=task, compute_loss=True)
        else:
            loss = model(batch, task=task, compute_loss=True)

        n_loss_units = loss.size(0)
        loss = loss.mean()
        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        if fp16:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
    return loss, n_loss_units
