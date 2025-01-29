import subprocess

import pytest

from srb.utils.isaacsim import get_isaacsim_python
from srb.utils.subprocess import terminate_process


def test_cli_ls():
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "ls",
        "--show_all",
    )

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if process.wait():
            stdout, stderr = process.communicate()
            pytest.fail(f"Process failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    except Exception as e:
        pytest.fail(f"Failed to start process\nException: {e}")
    finally:
        terminate_process(cmd, process)
