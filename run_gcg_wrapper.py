import signal
import runpy

# restore default SIGPIPE so Python does not raise/print BrokenPipeError on stdout close
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# run the script as __main__
runpy.run_path("run_gcg.py", run_name="__main__")