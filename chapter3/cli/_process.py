"""Shared logged subprocess runner for research experiments."""
from pathlib import Path
import subprocess


def run(command, log):
    final_model = None
    complete = False
    with log.open('x') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
            if line.startswith('Saved final trained model to '):
                final_model = Path(line.split(' to ', 1)[1].strip())
            if line.strip() == 'TRAINING_COMPLETE':
                complete = True
        code = process.wait()
    if code:
        raise RuntimeError('Command failed ({}), see {}'.format(code, log))
    return final_model, complete

