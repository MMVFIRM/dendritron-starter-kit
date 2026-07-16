from dendritron.cli import main


def test_info_command(capsys) -> None:
    assert main(["info"]) == 0
    assert "Dendritron primitive" in capsys.readouterr().out


def test_smoke_command(capsys) -> None:
    assert main(["smoke", "--json"]) == 0
    assert '"boolean"' in capsys.readouterr().out
