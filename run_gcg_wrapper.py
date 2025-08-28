import os
import sys
import signal

# restore default SIGPIPE so Python does not raise/print BrokenPipeError on stdout close
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# exec the target script so it becomes the main program image.
# This avoids problems when using the 'spawn' start method, which imports
# the main module by filename. Replacing the process with exec ensures
# child processes can import the module normally and avoids deadlocks
# caused by running the target as an ephemeral '__main__' via runpy.
# If user or environment hasn't set TQDM_MININTERVAL, set a conservative default
# to reduce the amount of progress-bar output written to SLURM logs.
os.environ.setdefault('TQDM_MININTERVAL', '10')
os.execv(sys.executable, [sys.executable, "run_gcg.py"] + sys.argv[1:])