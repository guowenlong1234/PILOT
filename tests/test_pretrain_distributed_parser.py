from pretrain_src.pretrain_src.parser import load_parser


def test_local_rank_defaults_to_torchrun_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")

    args = load_parser().parse_args(["--config", "unused.json"])

    assert args.local_rank == 1


def test_explicit_local_rank_overrides_torchrun_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")

    args = load_parser().parse_args(
        ["--config", "unused.json", "--local-rank", "0"]
    )

    assert args.local_rank == 0
