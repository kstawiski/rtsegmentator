#!/usr/bin/env python3
"""Create the Choi anatomical evidence report and clinician review sheet."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from choi_anatomy import main
if __name__ == '__main__':
    main()
