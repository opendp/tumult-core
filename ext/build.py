import subprocess
import sys
import platform
from pathlib import Path

build_dir = Path(__file__).parent
build_command = ["bash", str(build_dir / "build.sh")]

SUPPORTED_PLATFORMS = ["Linux", "Darwin"]
SUPPORTED_ARCHITECTURES = ["x86_64", "arm64"]


def check_platform():
    failed = False
    if platform.system() not in SUPPORTED_PLATFORMS:
        print(
            "It looks like you're running on an unsupported platform "
            f"({platform.system()}). Supported platforms are {SUPPORTED_PLATFORMS}."
        )
        failed = True
    elif platform.machine() not in SUPPORTED_ARCHITECTURES:
        print(
            "It looks like you're running on an unsupported architecture "
            f"('{platform.machine()}'). Supported architectures: "
            f"{SUPPORTED_ARCHITECTURES}"
        )
        failed = True
    if failed:
        print(
            "Contact us on slack at tmltdev.slack.com if you want help or to request "
            "support for your environment."
        )
        print("Here is more information about your system:")
        print(platform.uname())
        sys.exit(1)
    print(f"Running on: {platform.system()} {platform.machine()}")


check_platform()

try:
    subprocess.run(build_command, check=True)
except subprocess.CalledProcessError:
    print("=" * 80)
    print("Failed to build C dependencies, see above output for details.")
    print("=" * 80)
    sys.exit(1)
