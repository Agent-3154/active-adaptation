from active_adaptation.utils.wandb import _find_local_checkpoint


def test_latest_checkpoint_ignores_stale_pointer(tmp_path):
    (tmp_path / 'checkpoint_9.pt').touch()
    latest = tmp_path / 'checkpoint_100.pt'
    latest.touch()
    (tmp_path / 'last_checkpoint.txt').write_text('checkpoint_9.pt')
    assert _find_local_checkpoint(tmp_path, None) == latest
    assert _find_local_checkpoint(tmp_path, 9) == tmp_path / 'checkpoint_9.pt'
    assert _find_local_checkpoint(tmp_path, 10) is None


def test_final_checkpoint_wins_and_empty_directory_returns_none(tmp_path):
    assert _find_local_checkpoint(tmp_path, None) is None
    (tmp_path / 'checkpoint_invalid.pt').touch()
    (tmp_path / 'checkpoint_2000.pt').touch()
    final = tmp_path / 'checkpoint_final.pt'
    final.touch()
    assert _find_local_checkpoint(tmp_path, None) == final
