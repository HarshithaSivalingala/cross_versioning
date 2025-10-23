import tempfile
import os
import pytest

from src.dependency_upgrader import DependencyUpdater


class StubResolver:
    def __init__(self, mapping, requires=None):
        self.mapping = {k.lower(): v for k, v in mapping.items()}
        self.requires = {}
        if requires:
            for (name, version), spec in requires.items():
                key = (name.lower(), version)
                value = spec if isinstance(spec, list) else [spec]
                self.requires[key] = value

    def latest_version(self, package: str):
        return self.mapping.get((package or "").lower())

    def requires_dist(self, package: str, version: str):
        return self.requires.get(((package or "").lower(), version), [])

class TestDependencyUpdater:
    
    def test_update_requirements_txt(self):
        """Test requirements.txt updating"""
        resolver = StubResolver({
            "tensorflow": "2.15.0",
            "numpy": "1.24.0",
        })
        updater = DependencyUpdater(version_resolver=resolver)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            req_file = os.path.join(temp_dir, "requirements.txt")
            
            # Create sample requirements.txt
            with open(req_file, 'w') as f:
                f.write("tensorflow==1.15.0\n")
                f.write("numpy==1.18.0\n")
                f.write("requests==2.25.0\n")
            
            # Update dependencies
            success = updater.update_requirements_txt(temp_dir)
            
            assert success == True
            assert len(updater.updated_deps) > 0
            
            # Check updated content
            with open(req_file, 'r') as f:
                content = f.read()
            
            assert "tensorflow==2.15.0" in content
            assert "numpy==1.24.0" in content
            assert "requests==2.25.0" in content  # Non-ML library unchanged
    
    def test_update_nonexistent_requirements(self):
        """Test handling of non-existent requirements.txt"""
        updater = DependencyUpdater()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = updater.update_requirements_txt(temp_dir)
            assert success == False

    def test_update_nested_requirements(self):
        resolver = StubResolver({"tensorflow": "2.15.0"})
        updater = DependencyUpdater({"tensorflow": "2.15.0"}, version_resolver=resolver)

        with tempfile.TemporaryDirectory() as temp_dir:
            nested = os.path.join(temp_dir, "project", "configs")
            os.makedirs(nested)
            req_file = os.path.join(nested, "requirements.txt")
            with open(req_file, "w") as handle:
                handle.write("tensorflow==1.15.0\n")

            success = updater.update_requirements_txt(temp_dir)
            assert success is True

            with open(req_file, "r") as handle:
                contents = handle.read()
            assert "tensorflow==2.15.0" in contents

    def test_targeted_override_only_updates_specific_dependency(self):
        """Ensure overrides only modify the chosen dependency."""
        resolver = StubResolver(
            {"numpy": "1.24.0"},
            requires={
                ("tensorflow", "2.15.0"): ["numpy>=1.24.0"],
            },
        )
        updater = DependencyUpdater({"tensorflow": "2.15.0"}, version_resolver=resolver)

        with tempfile.TemporaryDirectory() as temp_dir:
            req_file = os.path.join(temp_dir, "requirements.txt")

            with open(req_file, "w") as f:
                f.write("tensorflow==1.15.0\n")
                f.write("numpy==1.18.0\n")

            success = updater.update_requirements_txt(temp_dir)
            assert success is True

            with open(req_file, "r") as f:
                content = f.read()

            assert "tensorflow==2.15.0" in content
            assert "numpy>=1.24.0" in content
