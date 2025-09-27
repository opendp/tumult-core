import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

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


class CustomBuildHook(BuildHookInterface):
    def initialize(self, _version: str, build_data: dict[str, Any]):
        # Tell hatchling to indicate that the package is not pure Python in its
        # metadata, and override the default wheel tag. Note that the infer_tag
        # option does not work for us here -- the wheel is not
        # Python-version-dependent, but infer_tag will assume that it only
        # supports the Python version being used to build it. Also note that due
        # to some disagreement between sysconfig and various packaging tools, we
        # need to slightly alter the format of the platform tag for it to be
        # accepted by cibuildwheel.
        build_data["pure_python"] = False
        build_data["tag"] = (
            f'py3-none-{sysconfig.get_platform().replace("-", "_").replace(".", "_")}'
        )

        check_platform()
        try:
            subprocess.run(build_command, check=True)
        except subprocess.CalledProcessError:
            print("=" * 80)
            print("Failed to build C dependencies, see above output for details.")
            print("=" * 80)
            sys.exit(1)
