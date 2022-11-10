import subprocess
import sys
from pathlib import Path

build_dir = Path(__file__).parent
build_command = ["bash", str(build_dir / "build.sh")]
try:
    subprocess.run(build_command, check=True)
except subprocess.CalledProcessError:
    print("=" * 80)
    print("Failed to build C dependencies, see above output for details.")
    print("=" * 80)
    sys.exit(1)
