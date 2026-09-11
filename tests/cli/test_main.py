from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points

import iivs_cli


def test_console_script_resolves_to_a_callable():
    # The wiring in `[project.scripts]` is only exercised when someone runs the
    # installed command, so a rename here fails for a user and for nobody else.
    scripts = entry_points(group="console_scripts")
    iivs = next(script for script in scripts if script.name == "iivs")
    assert iivs.value == "iivs_cli.main:main"
    assert callable(iivs.load())


def test_package_exports_the_entry_point():
    assert iivs_cli.__all__ == ("main",)
    assert callable(iivs_cli.main)


def test_importing_the_library_does_not_pull_in_the_cli():
    # The command ships beside the library, not inside it: `import iivs` must not
    # reach `iivs_cli`, or the split earns nothing. Checked in a fresh interpreter,
    # since this one has already imported both.
    source = (
        "import iivs;"
        "import sys;"
        "print([name for name in sys.modules if name.startswith('iivs_cli')])"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
