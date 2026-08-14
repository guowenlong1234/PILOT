from scripts.filter_habitat_startup_noise import filter_segment


NOISE = (
    b"PluginManager::Manager: duplicate static plugin "
    b"StbImageImporter, ignoring"
)


def test_drops_standalone_plugin_noise_and_its_newline():
    assert filter_segment(NOISE, b"\n") == b""


def test_removes_noise_appended_to_tqdm_progress_without_fixing_a_log_line():
    progress = b"  2%|#         | 4/200 [00:51<41:10, 12.61s/it, iter=4/200]"

    assert filter_segment(progress + NOISE, b"\n") == progress + b"\r"


def test_preserves_unrelated_output_exactly():
    warning = b"PluginManager::Manager: failed to load StbImageImporter"

    assert filter_segment(warning, b"\n") == warning + b"\n"


def test_removes_noise_without_discarding_adjacent_non_progress_text():
    prefix = b"worker-1: "

    assert filter_segment(prefix + NOISE, b"\n") == prefix + b"\n"
