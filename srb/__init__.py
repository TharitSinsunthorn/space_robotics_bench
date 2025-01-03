from srb.utils.tracing import with_rich

# Enable rich traceback
with_rich()

# Early import check
try:
    import omni as _  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"The Space Robotics Bench requires an environment with NVIDIA Omniverse and Isaac Sim installed: {e}"
    )
