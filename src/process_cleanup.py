"""
src/process_cleanup.py
======================
Gracefully terminate child processes spawned during an experiment run.
"""

import os
import signal

try:
    import psutil
except ImportError:
    psutil = None


def cleanup_child_processes(timeout: float = 3.0) -> None:
    """
    Send SIGTERM to all child processes, wait *timeout* seconds, then
    SIGKILL any survivors.

    Parameters
    ----------
    timeout : float
        Seconds to wait for graceful shutdown before force-killing.
    """
    if psutil is None:
        print("[process_cleanup] psutil not installed; skipping child cleanup.")
        return

    current = psutil.Process(os.getpid())
    children = current.children(recursive=True)

    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(children, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if children:
        print(
            f"[process_cleanup] Cleaned up {len(children)} child process(es) "
            f"({len(alive)} force-killed)."
        )
