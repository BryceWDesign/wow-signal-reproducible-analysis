from wow_signal_analysis import DISPLAY_NAME, PROJECT_SLUG, __version__


def test_package_identity_is_stable() -> None:
    assert DISPLAY_NAME == "Reproducible Analysis of the Wow! Signal"
    assert PROJECT_SLUG == "wow-signal-reproducible-analysis"
    assert __version__ == "0.1.0"
