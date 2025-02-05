import math
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

HEADLESS: bool = True
TEST_VISUAL_ENVS: bool = True

MAX_NUM_ENVS: int = 256
MAX_DURATION: float = 20.0


def _list_envs() -> Iterable[str]:
    if envs := read_env_list_cache():
        if not TEST_VISUAL_ENVS:
            envs = filter(lambda env: not _is_visual_env(env), envs)
        envs = sorted(envs)
    else:
        test_filepath = Path(__file__)
        logging.warning(
            f"Skipping {test_filepath.parent.name}/{test_filepath.name} because no environments were found in the cache (repeat the test)"
        )
        envs = []

    return envs


def _is_visual_env(env: str) -> bool:
    return env.endswith("_visual")


@pytest.mark.parametrize(
    "env,num_envs",
    (
        (env, num_envs)
        for env in _list_envs()
        for num_envs in (1, math.floor(math.sqrt(MAX_NUM_ENVS)), MAX_NUM_ENVS)
        if (not _is_visual_env(env) or num_envs < MAX_NUM_ENVS)
    ),
)
def test_cli_agent_rand(env: str, num_envs: int):
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "agent",
        "rand",
        "--headless" if HEADLESS else "--hide_ui",
        f"env.scene.num_envs={num_envs}",
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
        while time.time() - start < MAX_DURATION:
            if process.poll() is not None:
                logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
                stdout, stderr = process.communicate()
                pytest.fail(
                    f'Process failed for env "{env}"\n[env={env}] STDOUT:\n{stdout}\n[env={env}] STDERR:\n{stderr}'
                )
            time.sleep(0.01 * MAX_DURATION)
    except Exception as e:
        logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
        pytest.fail(
            f'Failed to start process for env "{env}"\n[env={env}] Exception: {e}'
        )
    finally:
        terminate_process("srb", process)
