"""Build training datasets without requiring the package entry point."""

import sys

from energy.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "build-training-data":
        sys.argv.insert(1, "build-training-data")
    main()
