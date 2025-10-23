import json
import os
import re
from typing import Dict, Optional, Tuple
from urllib import error, request

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet


class PyPIVersionResolver:
    """Resolve package versions using PyPI's simple JSON API."""

    PYPI_URL_TEMPLATE = "https://pypi.org/pypi/{package}"

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self._json_cache: Dict[Tuple[str, Optional[str]], Optional[dict]] = {}
        self._latest_cache: Dict[str, Optional[str]] = {}

    def _fetch_json(self, package: str, version: Optional[str] = None) -> Optional[dict]:
        key = (package or "").strip().lower()
        if not key:
            return None

        cache_key = (key, version)
        if cache_key in self._json_cache:
            return self._json_cache[cache_key]

        base_url = self.PYPI_URL_TEMPLATE.format(package=key.replace("_", "-"))
        if version:
            url = f"{base_url}/{version}/json"
        else:
            url = f"{base_url}/json"

        try:
            with request.urlopen(url, timeout=self.timeout) as response:
                if response.status != 200:
                    self._json_cache[cache_key] = None
                    return None

                payload = response.read().decode("utf-8")
                data = json.loads(payload)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
            self._json_cache[cache_key] = None
            return None

        self._json_cache[cache_key] = data
        return data

    def latest_version(self, package: str) -> Optional[str]:
        key = (package or "").strip().lower()
        if not key:
            return None

        if key in self._latest_cache:
            return self._latest_cache[key]

        data = self._fetch_json(package)
        if not data:
            self._latest_cache[key] = None
            return None

        version = (data.get("info") or {}).get("version")
        self._latest_cache[key] = version
        return version

    def requires_dist(self, package: str, version: str) -> Optional[list]:
        data = self._fetch_json(package, version=version)
        if not data:
            return []

        info = data.get("info") or {}
        requires = info.get("requires_dist")
        if not requires:
            return []
        return requires


class DependencyUpdater:
    """Update repository dependencies using dynamic version lookups."""

    def __init__(
        self,
        target_versions: Optional[Dict[str, str]] = None,
        version_resolver: Optional[PyPIVersionResolver] = None,
    ):
        self.updated_deps = []
        self.target_versions = self._normalize_target_versions(target_versions)
        self.version_resolver = version_resolver or PyPIVersionResolver()
        # Cache resolved specs so setup.py updates stay consistent with requirements.txt
        self.resolved_versions: Dict[str, str] = dict(self.target_versions)
        self._populate_metadata_dependencies()

    @staticmethod
    def _normalize_version_spec(raw_spec: str) -> str:
        spec = (raw_spec or "").strip()
        if not spec:
            raise ValueError("Version specification cannot be empty.")

        if spec[0] not in (">", "<", "=", "!", "~"):
            spec = f"=={spec}"
        return spec

    def _normalize_target_versions(self, overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not overrides:
            return {}

        normalized: Dict[str, str] = {}
        for name, spec in overrides.items():
            key = (name or "").strip().lower()
            if not key:
                continue
            normalized[key] = self._normalize_version_spec(spec or "")
        return normalized

    def _target_version_for(self, package: str) -> Optional[str]:
        key = package.lower()
        if key in self.resolved_versions:
            return self.resolved_versions[key]

        latest = self.version_resolver.latest_version(package)
        if not latest:
            return None

        spec = self._normalize_version_spec(latest)
        self.resolved_versions[key] = spec
        return spec

    def _populate_metadata_dependencies(self) -> None:
        for package, spec in self.target_versions.items():
            version = self._exact_version_for_metadata(package, spec)
            if not version:
                continue

            requires_dist = self.version_resolver.requires_dist(package, version)
            if not requires_dist:
                continue

            for requirement_str in requires_dist:
                try:
                    requirement = Requirement(requirement_str)
                except InvalidRequirement:
                    continue

                if requirement.marker and not requirement.marker.evaluate():
                    continue

                specifier = str(requirement.specifier).strip()
                if not specifier:
                    continue

                normalized_name = requirement.name.lower()
                if normalized_name in self.target_versions:
                    # Preserve explicit user overrides
                    continue

                self.resolved_versions.setdefault(normalized_name, specifier)

    def _exact_version_for_metadata(self, package: str, spec: str) -> Optional[str]:
        normalized_spec = (spec or "").strip()
        if not normalized_spec:
            return None

        if normalized_spec.startswith("==") and "," not in normalized_spec:
            return normalized_spec[2:]

        try:
            spec_set = SpecifierSet(normalized_spec)
        except InvalidSpecifier:
            return None

        latest = self.version_resolver.latest_version(package)
        if latest and spec_set.contains(latest):
            return latest

        return None

    def _update_single_requirements_file(self, path: str) -> bool:
        with open(path, "r") as handle:
            lines = handle.readlines()

        updated_any = False
        updated_lines = []
        for line in lines:
            raw_line = line.rstrip("\n")
            stripped = raw_line.strip()

            if not stripped or stripped.startswith("#"):
                updated_lines.append(raw_line + "\n")
                continue

            pkg_match = re.match(r"^([a-zA-Z0-9_.+-]+)", stripped)
            if pkg_match:
                original_pkg = pkg_match.group(1)
                new_spec = self._target_version_for(original_pkg)
                if new_spec:
                    new_line = f"{original_pkg}{new_spec}"
                    if new_line != stripped:
                        updated_lines.append(new_line + "\n")
                        self.updated_deps.append(f"{stripped} → {new_line}")
                        updated_any = True
                        continue

            updated_lines.append(raw_line + "\n")

        if not updated_any:
            return False

        with open(path, "w") as handle:
            handle.writelines(updated_lines)

        return True

    def update_requirements_txt(self, repo_path: str) -> bool:
        """Update requirements.txt entries throughout the repository."""
        any_found = False
        any_updated = False

        for root, _, files in os.walk(repo_path):
            if "requirements.txt" in files:
                any_found = True
                file_path = os.path.join(root, "requirements.txt")
                updated = self._update_single_requirements_file(file_path)
                any_updated = any_updated or updated

        if not any_found:
            return False

        return any_updated

    def update_setup_py(self, repo_path: str) -> bool:
        """Update setup.py install_requires style dependencies."""
        setup_path = os.path.join(repo_path, "setup.py")
        if not os.path.exists(setup_path):
            return False

        with open(setup_path, "r") as handle:
            content = handle.read()

        pattern = re.compile(
            r'(["\'])([a-zA-Z0-9_.+-]+)([><=!~]=?[^"\',]+)(["\'])',
            flags=re.IGNORECASE,
        )

        def _replace(match: re.Match[str]) -> str:
            open_quote, package, existing_spec, close_quote = match.groups()
            new_spec = self._target_version_for(package)
            if not new_spec:
                return match.group(0)

            replacement = f"{open_quote}{package}{new_spec}{close_quote}"
            if replacement != match.group(0):
                normalized_key = package.lower()
                self.resolved_versions[normalized_key] = new_spec
                self.updated_deps.append(f"{package}{existing_spec} → {package}{new_spec}")
            return replacement

        updated_content, replacements = pattern.subn(_replace, content)

        if replacements == 0 or updated_content == content:
            return False

        with open(setup_path, "w") as handle:
            handle.write(updated_content)

        return True
