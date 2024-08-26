# Adapted from https://github.com/pytorch/pytorch/blob/eb15b1a016c6facaf8605dde2c20b5de1586542d/torch/multiprocessing/spawn.py

import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import sys
import tempfile
import time
import warnings
from typing import Optional

import torch.distributed

# from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]

log = logging.getLogger(__name__)


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return type(self), (self.msg, self.error_index, self.pid)


class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
    ):
        super().__init__(msg, error_index, error_pid)


class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""

    __slots__ = ["exit_code"]

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: Optional[str] = None,
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (
            type(self),
            (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name),
        )


def _wrap(fn, port, rank, nprocs, args, error_file):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    # _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        try:
            import torch.distributed as dist
            dist.init_process_group(backend="nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=nprocs)
        except:
            warnings.warn("Need to manually initalize distributed environment")
        fn(*args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback

        with open(error_file, "wb") as fh:
            pickle.dump(traceback.format_exc(), fh)
        sys.exit(1)


class ProcessContext:
    def __init__(self, processes, error_files):
        self.error_files = error_files
        self.processes = processes
        self.sentinels = {
            process.sentinel: index for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        r"""Join one or more processes within spawn context.

        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0

        # Assume failure. Terminate processes that are still alive.
        # Try SIGTERM then SIGKILL if the process isn't going down.
        # The reason is related to python signal handling is limited
        # to main thread and if that is in c/c++ land and stuck it won't
        # to handle it. We have seen processes getting stuck not handling
        # SIGTERM for the above reason.
        timeout: int = 30
        for process in self.processes:
            if process.is_alive():
                log.warning("Terminating process %s via signal SIGTERM", process.pid)
                process.terminate()
        end = time.monotonic() + timeout
        for process in self.processes:
            time_to_wait = max(0, end - time.monotonic())
            process.join(time_to_wait)
        for process in self.processes:
            if process.is_alive():
                log.warning(
                    "Unable to shutdown process %s via SIGTERM , forcefully exiting via SIGKILL",
                    process.pid,
                )
                process.kill()
            process.join()

        # The file will only be created if the process crashed.
        failed_process = self.processes[error_index]
        if not os.access(self.error_files[error_index], os.R_OK):
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                try:
                    name = signal.Signals(-exitcode).name
                except ValueError:
                    name = f"<Unknown signal {-exitcode}>"
                raise ProcessExitedException(
                    "process %d terminated with signal %s" % (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name,
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" % (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                )

        with open(self.error_files[error_index], "rb") as fh:
            original_trace = pickle.load(fh)
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_files):
        warnings.warn("SpawnContext is renamed to ProcessContext since 1.4 release.")
        super().__init__(processes, error_files)


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(
    fn, port, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"
):
    mp = multiprocessing.get_context(start_method)
    error_files = []
    processes = []
    for rank in range(nprocs):
        # Each process is assigned a file to write tracebacks to.  We
        # use the file being non-empty to indicate an exception
        # occurred (vs an expected shutdown).  Note: this previously
        # used a multiprocessing.Queue but that can be prone to
        # deadlocks, so we went with a simpler solution for a one-shot
        # message between processes.
        tf = tempfile.NamedTemporaryFile(
            prefix="pytorch-errorfile-", suffix=".pickle", delete=False
        )
        tf.close()
        os.unlink(tf.name)
        process = mp.Process(
            target=_wrap,
            args=(fn, port, rank, nprocs, args, tf.name),
            daemon=daemon,
        )
        process.start()
        error_files.append(tf.name)
        processes.append(process)

    context = ProcessContext(processes, error_files)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn", port=29600):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != "spawn":
        msg = (
            "This method only supports start_method=spawn (got: %s).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)" % start_method
        )
        warnings.warn(msg)
    return start_processes(fn, port, args, nprocs, join, daemon, start_method="spawn")
