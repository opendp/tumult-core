import subprocess
from pathlib import Path

build_dir = Path(__file__).parent
build_command = ["bash", str(build_dir / "build.sh")]
subprocess.run(build_command, check=True)
