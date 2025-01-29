import os
import subprocess
import time
from pathlib import Path

import psutil
import pytest

from srb.utils import logging
from srb.utils.path import SRB_ENV_CACHE_PATH


def read_envs():
    """Reads the list of environments from the .envs_cache file."""
    if not SRB_ENV_CACHE_PATH.exists():
        pytest.fail(f"Environment cache file '{SRB_ENV_CACHE_PATH}' not found.")
    with SRB_ENV_CACHE_PATH.open("r") as f:
        return f.read().splitlines()


@pytest.mark.parametrize("env_name", read_envs())
def test_rand_agent_for_env(env_name):
    """Runs the SRB agent on the environment and ensures termination."""

    if isaac_sim_python := os.environ.get("ISAAC_SIM_PYTHON"):
        python_path = Path(isaac_sim_python)
    else:
        python_path = (
            Path(os.environ.get("HOME", "/root"))
            .joinpath("isaac-sim")
            .joinpath("python.sh")
        )
    if not python_path.exists():
        python = "python3"
    else:
        python = python_path.as_posix()

    cmd = [python, "-m", "srb", "agent", "rand", "--hide_ui", "--env", env_name]
    logging.info(f"Starting process for '{env_name}' with command: {cmd}")

    try:
        # Start the subprocess in its own process group
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        logging.info(f"Process for '{env_name}' started with PID: {process.pid}")

        # Let it run for a while
        time.sleep(30)

        # Check if process exited early
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"Process for '{env_name}' exited early:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

    except Exception as e:
        pytest.fail(f"Failed to start process for '{env_name}': {e}")

    finally:
        terminate_process(process)


def terminate_process(process):
    """Terminates the process using `pkill` and `psutil`."""
    logging.info(f"Attempting to terminate process {process.pid}...")

    try:
        # First attempt: use pkill to kill all matching processes
        logging.info("Using `pkill` to forcefully stop `srb agent rand`...")
        subprocess.run(["pkill", "-9", "-f", "srb agent rand"], check=False)

        # Wait a moment to let processes exit
        time.sleep(3)

        # Second attempt: use psutil for additional cleanup
        if psutil.pid_exists(process.pid):
            parent = psutil.Process(process.pid)

            # Terminate children first
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()

            time.sleep(3)  # Give time to exit

            # If still running, escalate to SIGKILL
            for child in parent.children(recursive=True):
                if child.is_running():
                    child.kill()
            if parent.is_running():
                parent.kill()

    except psutil.NoSuchProcess:
        pass  # Already terminated

    except Exception as e:
        logging.error(f"Error while terminating process {process.pid}: {e}")
