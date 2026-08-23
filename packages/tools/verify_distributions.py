from __future__ import annotations

import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement

EXPECTED_WHEELS = {
    "uod_rg24_contracts": "contracts",
    "uod_rg24_preprocessing": "preprocessing",
    "uod_rg24_runtime_azure": "runtime-azure",
}
PACKAGES_ROOT = Path(__file__).resolve().parents[1]


def _distribution_key(wheel: Path) -> str:
    for key in EXPECTED_WHEELS:
        if wheel.name.startswith(f"{key}-"):
            return key
    raise AssertionError(f"unexpected wheel: {wheel.name}")


def _source_payloads(key: str) -> dict[str, bytes]:
    source = PACKAGES_ROOT / EXPECTED_WHEELS[key] / "src" / key
    payloads: dict[str, bytes] = {}
    for path in source.rglob("*"):
        if path.is_file() and (
            path.suffix in {".py", ".json"} or path.name == "py.typed"
        ):
            archive_name = f"{key}/{path.relative_to(source).as_posix()}"
            payloads[archive_name] = path.read_bytes()
    return payloads


def _runtime_requirements(raw_requirements: list[str]) -> dict[str, Requirement]:
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, raw_requirements)
    }
    assert len(requirements) == len(raw_requirements), "duplicate runtime requirement"
    return requirements


def verify(dist_directory: Path) -> None:
    wheels = sorted(dist_directory.glob("*.whl"))
    sdists = sorted(dist_directory.glob("*.tar.gz"))
    assert len(wheels) == 3, f"expected 3 wheels, found {len(wheels)}"
    assert len(sdists) == 3, f"expected 3 sdists, found {len(sdists)}"
    assert {_distribution_key(wheel) for wheel in wheels} == set(EXPECTED_WHEELS)

    for wheel in wheels:
        key = _distribution_key(wheel)
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            assert f"{key}/py.typed" in names, f"{wheel.name} omits py.typed"

            expected_payloads = _source_payloads(key)
            wheel_payloads = {
                name
                for name in names
                if name.startswith(f"{key}/")
                and (
                    Path(name).suffix in {".py", ".json"} or name.endswith("/py.typed")
                )
            }
            assert wheel_payloads == set(expected_payloads), (
                f"{wheel.name} package contents differ from source"
            )
            for name, expected in expected_payloads.items():
                assert archive.read(name) == expected, (
                    f"{wheel.name}:{name} differs from source"
                )

            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            requirements = metadata.get_all("Requires-Dist", [])

            if key == "uod_rg24_contracts":
                schemas = {
                    name
                    for name in names
                    if name.startswith("uod_rg24_contracts/schemas/v1/")
                    and name.endswith(".json")
                }
                assert len(schemas) == 8, f"expected 8 schemas, found {len(schemas)}"

            if key == "uod_rg24_runtime_azure":
                parsed = _runtime_requirements(requirements)
                for internal in ("uod-rg24-contracts", "uod-rg24-preprocessing"):
                    requirement = parsed[internal]
                    assert str(requirement.specifier) == "==0.1.0"
                    assert requirement.url is None
                assert str(parsed["pyarrow"].specifier) == "==25.0.0"
                lowered = "\n".join(requirements).lower()
                for forbidden in ("file:", "git+", " @ ", "../", "\\"):
                    assert forbidden not in lowered, (
                        f"runtime wheel contains a local/VCS requirement: {forbidden}"
                    )
                layouts = {
                    name
                    for name in names
                    if name.startswith("uod_rg24_runtime_azure/layouts/v1/")
                    and name.endswith(".json")
                }
                assert len(layouts) == 5, (
                    f"expected 5 runtime layout descriptors, found {len(layouts)}"
                )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_distributions.py <dist-directory>")
    verify(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
