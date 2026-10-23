"""Allow `python -m modelbench`."""

import sys

from .cli import main

sys.exit(main())
