"""Start Curio's local web UI with the bundled catalog and a user-writable cache."""
from pathlib import Path
import sys
import threading
import time
import urllib.request
import webbrowser

def open_when_ready():
    for _ in range(120):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8765/', timeout=1) as r:
                if r.status == 200:
                    webbrowser.open('http://127.0.0.1:8765/')
                    return
        except OSError:
            time.sleep(.5)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    from curio.cli import main
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
    smoke = '--smoke-test' in sys.argv
    if not smoke:
        print('Curio: keep this window open. First identification downloads model weights.')
        threading.Thread(target=open_when_ready, daemon=True).start()
    args = ['--packs', str(root / 'packs')]
    if smoke:
        args += ['--embedder', 'simple']
    main(args + ['serve'])
