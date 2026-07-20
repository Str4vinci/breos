"""Build and smoke-test BREOS release artifacts."""

from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import textwrap
import venv
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_WHEEL_FILES = {
    "breos/data/configs/locations.json",
    "breos/data/configs/costs.json",
    "breos/data/configs/electricity.json",
    "breos/data/configs/emissions.json",
    "breos/data/configs/financials.json",
    "breos/data/rlp/h0SLP_demandlib_1000kwh_hourly.csv",
    "breos/data/rlp/h0SLP_demandlib_1000kwh_15min.csv",
    "breos/degradation/blast/LICENSE",
    "breos/degradation/blast/NOTICE",
}
REQUIRED_BLAST_MODEL_FILES = {
    "lfp_gr_250AhPrismatic_2019.py",
    "lfp_gr_SonyMurata3Ah_2018.py",
    "lmo_gr_NissanLeaf66Ah_2ndLife_2020.py",
    "nca_gr_Panasonic3Ah_2018.py",
    "nca_grsi_SonyMurata2p5Ah_2023.py",
    "nmc111_gr_Kokam75Ah_2017.py",
    "nmc111_gr_Sanyo2Ah_2014.py",
    "nmc622_gr_DENSO50Ah_2021.py",
    "nmc811_grSi_LGM50_5Ah_2021.py",
    "nmc811_grSi_LGMJ1_4Ah_2020.py",
    "nmc_gr_50Ah_B1_2020.py",
    "nmc_gr_50Ah_B2_2020.py",
    "nmc_gr_75Ah_A_2019.py",
    "nmc_lto_10Ah_2020.py",
}
REQUIRED_WHEEL_FILES.update(f"breos/degradation/blast/models/{filename}" for filename in REQUIRED_BLAST_MODEL_FILES)
REQUIRED_SDIST_FILES = {
    "docs/conf.py",
    "docs/index.md",
    "docs/release.md",
    "docs/getting-started/quickstart.md",
    "docs/legal/load-profile-data.md",
}


def _run(
    command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        sys.stdout.write(exc.stdout or "")
        sys.stderr.write(exc.stderr or "")
        raise


def _build_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build release artifacts")

    result = _run([uv, "build", "--out-dir", str(dist_dir)])
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        raise AssertionError(f"Expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        raise AssertionError(f"Expected exactly one sdist, found {len(sdists)}")
    return wheels[0], sdists[0]


def _assert_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    missing = sorted(REQUIRED_WHEEL_FILES - names)
    if missing:
        raise AssertionError(f"Wheel is missing packaged runtime data: {missing}")
    leaked_build_docs = sorted(name for name in names if name.startswith("docs/_build/") or "/docs/_build/" in name)
    if leaked_build_docs:
        raise AssertionError(f"Wheel contains generated docs: {leaked_build_docs[:5]}")


def _sdist_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist, "r:gz") as archive:
        names = set()
        for member in archive.getmembers():
            path = Path(member.name)
            try:
                names.add(path.relative_to(path.parts[0]).as_posix())
            except ValueError:
                names.add(path.as_posix())
        return names


def _assert_sdist_contents(sdist: Path) -> None:
    names = _sdist_names(sdist)
    missing = sorted(REQUIRED_SDIST_FILES - names)
    if missing:
        raise AssertionError(f"Sdist is missing rebuildable docs source: {missing}")
    leaked_build_docs = sorted(name for name in names if name.startswith("docs/_build/"))
    if leaked_build_docs:
        raise AssertionError(f"Sdist contains generated docs: {leaked_build_docs[:5]}")


def _dependency_paths() -> str:
    paths = set()
    try:
        paths.update(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        paths.add(user_site)
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        paths.add(purelib)
    return os.pathsep.join(str(Path(path).resolve()) for path in paths if Path(path).exists())


def _venv_paths(venv_dir: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe", venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "python", venv_dir / "bin" / "pip"


def _smoke_test_installed_wheel(wheel: Path, work_dir: Path) -> None:
    venv_dir = work_dir / "installed-wheel"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python, pip = _venv_paths(venv_dir)

    install = _run([str(pip), "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=work_dir)
    sys.stdout.write(install.stdout)
    sys.stderr.write(install.stderr)

    smoke_code = textwrap.dedent(
        """
        import json
        import os
        import sys
        from importlib.resources import files
        from pathlib import Path

        repo_root = Path(os.environ["BREOS_REPO_ROOT"]).resolve()
        dep_paths = [p for p in os.environ.get("BREOS_DEPENDENCY_PATHS", "").split(os.pathsep) if p]

        sys.path = [
            p
            for p in sys.path
            if Path(p or ".").resolve() != repo_root
            and repo_root not in Path(p or ".").resolve().parents
        ]
        for dep_path in dep_paths:
            if dep_path not in sys.path:
                sys.path.append(dep_path)

        import breos
        from breos.app import App
        from breos.degradation.engine import BLAST_MODEL_CLASSES
        from breos.degradation.profiles import BLAST_UPSTREAM_COMMIT, BLAST_UPSTREAM_VERSION
        from breos.load_profiles import load_profile
        from breos.resources import load_config_json, rlp_resource

        breos_file = Path(breos.__file__).resolve()
        if repo_root == breos_file or repo_root in breos_file.parents:
            raise AssertionError(f"Imported breos from source tree: {breos_file}")

        locations = load_config_json("locations.json")
        if "porto" not in locations:
            raise AssertionError("packaged locations.json did not contain porto")

        hourly = rlp_resource("h0SLP_demandlib_1000kwh_hourly.csv")
        if not hourly.is_file():
            raise AssertionError(f"packaged RLP file is missing: {hourly}")

        profile = load_profile("1", 1000, start_date="2025-01-01", freq="h", timezone="UTC")
        if len(profile) != 8760:
            raise AssertionError(f"unexpected hourly load profile length: {len(profile)}")

        app = App({"location": "porto", "n_modules": 1, "annual_consumption_kwh": 1000})
        if app._cfg["location"] != "porto":
            raise AssertionError("App did not resolve packaged configuration")

        profiles = breos.list_battery_models()
        if len(profiles) != 14 or set(BLAST_MODEL_CLASSES) != {profile["key"] for profile in profiles}:
            raise AssertionError("installed wheel does not expose all 14 BLAST profiles and model classes")
        for model_key, model_class in BLAST_MODEL_CLASSES.items():
            model = model_class()
            if "q" not in model.outputs:
                raise AssertionError(f"installed BLAST model {model_key} has no capacity output")
        if (
            BLAST_UPSTREAM_VERSION != "1.1.0"
            or BLAST_UPSTREAM_COMMIT != "d789e00bca60f628de640745c18eb724b07358bd"
        ):
            raise AssertionError("installed BLAST provenance does not match the vendored pin")

        blast_package = files("breos.degradation.blast")
        license_text = blast_package.joinpath("LICENSE").read_text(encoding="utf-8")
        notice_text = blast_package.joinpath("NOTICE").read_text(encoding="utf-8")
        if "Copyright (c) 2023, Alliance for Sustainable Energy" not in license_text:
            raise AssertionError("installed BLAST license is missing or unexpected")
        if "Alliance for Sustainable Energy" not in notice_text:
            raise AssertionError("installed BLAST DOE attribution notice is missing or unexpected")

        print(
            json.dumps(
                {
                    "breos_file": str(breos_file),
                    "profile_rows": len(profile),
                    "blast_models": len(profiles),
                    "blast_upstream": f"{BLAST_UPSTREAM_VERSION}@{BLAST_UPSTREAM_COMMIT}",
                }
            )
        )
        """
    )
    env = os.environ.copy()
    env["BREOS_REPO_ROOT"] = str(REPO_ROOT)
    env["BREOS_DEPENDENCY_PATHS"] = _dependency_paths()
    smoke = _run([str(python), "-c", smoke_code], cwd=work_dir, env=env)
    sys.stdout.write(smoke.stdout)
    sys.stderr.write(smoke.stderr)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="breos-release-") as tmp:
        work_dir = Path(tmp)
        dist_dir = work_dir / "dist"
        dist_dir.mkdir()

        wheel, sdist = _build_artifacts(dist_dir)
        _assert_wheel_contents(wheel)
        _assert_sdist_contents(sdist)
        _smoke_test_installed_wheel(wheel, work_dir)

    print("Release artifact verification passed.")


if __name__ == "__main__":
    main()
