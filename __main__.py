"""
qweave -- command-line entry.

The full CLI lives in phase1_basic.cli so that the installed console
script `qweave` and `python __main__.py` share one code path.
"""

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_basic.cli import main

if __name__ == "__main__":
    main()
