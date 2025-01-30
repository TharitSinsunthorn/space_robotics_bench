import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

import pytest

from srb.utils import logging
from srb.utils.cache import read_env_list_cache
from srb.utils.isaacsim import get_isaacsim_python
from srb.utils.subprocess import terminate_process

DURATION: float = 30.0
HEADLESS: bool = True
NUM_ENVS: int = 2
TEST_VISUAL_ENVS: bool = True


def list_envs() -> Iterable[str] | None:
    envs = read_env_list_cache()

    if not envs:
        test_filepath = Path(__file__)
        logging.warning(
            f"Skipping {test_filepath.parent.name}/{test_filepath.name} because no environments were found in the cache (repeat the test)"
        )
    else:
        if not TEST_VISUAL_ENVS:
            envs = filter(lambda env: not env.endswith("_visual"), envs)
        envs = sorted(envs)

    return envs


@pytest.mark.order(after="cli_ls_test.py::test_cli_ls")
@pytest.mark.parametrize("env", list_envs() or [])
def test_cli_agent_rand(env: str):
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "agent",
        "rand",
        "--headless" if HEADLESS else "--hide_ui",
        f"env.scene.num_envs={NUM_ENVS}",
        "--env",
        env,
    )

    environ = os.environ.copy()
    environ["SF_BAKER"] = "0"

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        start = time.time()
        while time.time() - start < DURATION:
            if process.poll() is not None:
                logging.critical(f'[{env}] Failed command: {" ".join(cmd)}')
                stdout, stderr = process.communicate()
                pytest.fail(
                    f'Process failed for env "{env}"\n[env={env}] STDOUT:\n{stdout}\n[env={env}] STDERR:\n{stderr}'
                )
            time.sleep(0.1)
    except Exception as e:
        logging.critical(f'[{env}] Failed command: {" ".join(cmd)}')
        pytest.fail(
            f'Failed to start process for env "{env}"\n[env={env}] Exception: {e}'
        )
    finally:
        terminate_process("srb", process)
