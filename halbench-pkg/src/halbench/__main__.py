"""Allow `python -m halbench ...` in addition to the `halbench` script."""
from halbench.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
