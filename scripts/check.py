"""Run unit tests and the official course tests."""

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "70049809ec08bf1e0ac2f4c673c43939072cfe90"


def main():
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=True,
    )
    tests = ROOT / ".cache/official"
    tests.mkdir(parents=True, exist_ok=True)
    for name in ["conftest.py", "test_project1_public.py"]:
        if not (tests / name).exists():
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/epfml/ML_course/{COMMIT}/projects/project1/grading_tests/{name}",
                tests / name,
            )
    with tempfile.TemporaryDirectory(prefix="ml1-check-") as folder:
        for source in list(ROOT.glob("*.py")) + [ROOT / "README.md"]:
            shutil.copy2(source, folder)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(tests),
                "--rootdir",
                str(ROOT),
                "--github_link",
                folder,
                "-k",
                "not test_github_link_format",
                "-q",
            ],
            cwd=ROOT,
            check=True,
        )
    print("Local tests passed.")


if __name__ == "__main__":
    main()
