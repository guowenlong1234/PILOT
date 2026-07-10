def step_amp_optimizer(scaler, optimizer, scheduler=None):
    """Step an AMP optimizer and advance its scheduler only on success."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = scaler.get_scale() >= scale_before
    if optimizer_stepped and scheduler is not None:
        scheduler.step()
    return optimizer_stepped
