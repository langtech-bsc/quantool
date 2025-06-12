import os
import sys
import subprocess
import logging
import pytest
from unittest.mock import MagicMock
from quantool.utils.command import run_command

@pytest.fixture
def mock_logger():
    """Return a Logger mock."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Patch subprocess.run in our module."""
    mock_run = MagicMock()
    monkeypatch.setattr('quantool.utils.command.subprocess.run', mock_run)
    return mock_run

@pytest.fixture
def mock_sys_exit(monkeypatch):
    """Patch sys.exit to prevent exiting the test process."""
    mock_exit = MagicMock()
    monkeypatch.setattr('quantool.utils.command.sys.exit', mock_exit)
    return mock_exit

def test_run_command_success(mock_logger, mock_subprocess_run, monkeypatch):
    """stdout lines are logged, written, and returned; success message is logged."""
    fake = MagicMock()
    fake.stdout = "line1\nline2"
    fake.stderr = ""
    mock_subprocess_run.return_value = fake

    fake_stdout = MagicMock()
    fake_stderr = MagicMock()
    monkeypatch.setattr(sys, 'stdout', fake_stdout)
    monkeypatch.setattr(sys, 'stderr', fake_stderr)

    out = run_command(mock_logger, ["cmd", "arg"])

    # subprocess.run call
    mock_subprocess_run.assert_called_once_with(
        ["cmd", "arg"],
        cwd=os.path.join(os.getcwd(), "."),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    # logging & writes
    mock_logger.debug.assert_called_once()
    mock_logger.info.assert_any_call("line1")
    mock_logger.info.assert_any_call("line2")
    mock_logger.info.assert_any_call("Command 'cmd arg' completed successfully")
    fake_stdout.write.assert_any_call("line1\n")
    fake_stdout.write.assert_any_call("line2\n")
    assert out == "line1\nline2"

def test_run_command_with_stderr(mock_logger, mock_subprocess_run, monkeypatch):
    """stderr lines are logged as warnings and written to stderr."""
    fake = MagicMock()
    fake.stdout = "out"
    fake.stderr = "warn"
    mock_subprocess_run.return_value = fake

    fake_stdout = MagicMock()
    fake_stderr = MagicMock()
    monkeypatch.setattr(sys, 'stdout', fake_stdout)
    monkeypatch.setattr(sys, 'stderr', fake_stderr)

    out = run_command(mock_logger, ["cmd"])

    mock_logger.warning.assert_called_once_with("warn")
    fake_stdout.write.assert_any_call("out\n")
    fake_stderr.write.assert_any_call("warn\n")
    assert out == "out"

def test_run_command_failure(mock_logger, mock_subprocess_run, mock_sys_exit, monkeypatch):
    """On CalledProcessError, stdout/stderr are logged, exception is logged, exit(127)."""
    err = subprocess.CalledProcessError(1, ["fail"])
    err.stdout = "errout"
    err.stderr = "errmsg"
    mock_subprocess_run.side_effect = err

    fake_stdout = MagicMock()
    fake_stderr = MagicMock()
    monkeypatch.setattr(sys, 'stdout', fake_stdout)
    monkeypatch.setattr(sys, 'stderr', fake_stderr)

    run_command(mock_logger, ["fail"])

    mock_logger.info.assert_any_call("errout")
    mock_logger.warning.assert_any_call("errmsg")
    mock_logger.exception.assert_called_once()
    fake_stdout.write.assert_any_call("errout\n")
    fake_stderr.write.assert_any_call("errmsg\n")
    mock_sys_exit.assert_called_once_with(127)

def test_run_command_custom_cwd(mock_logger, mock_subprocess_run):
    """Custom cwd is passed through to subprocess.run."""
    fake = MagicMock()
    fake.stdout = ""
    fake.stderr = ""
    mock_subprocess_run.return_value = fake

    custom = "my/dir"
    run_command(mock_logger, ["cmd"], cwd=custom)

    _, kwargs = mock_subprocess_run.call_args
    assert kwargs["cwd"] == os.path.join(os.getcwd(), custom)

def test_run_command_real_echo():
    """Integration: real echo command returns its output."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())

    out = run_command(logger, ["echo", "Hello World"])
    assert "Hello World" in out