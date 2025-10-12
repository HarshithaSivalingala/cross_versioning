import os
import re
import ast
from typing import Set

class SmartDependencyUpdater:
    """Intelligently detects and updates USED dependencies in requirements.txt and setup.py"""
    
    ML_DEPENDENCIES = {
        'tensorflow': '>=2.15.0',
        'torch': '>=2.0.0',
        'torchvision': '>=0.15.0',
        'numpy': '>=1.24.0',
        'jax': '>=0.4.0',
        'jaxlib': '>=0.4.0',
        'scikit-learn': '>=1.3.0',
        'sklearn': '>=1.3.0',
        'pandas': '>=2.0.0',
        'matplotlib': '>=3.7.0',
        'seaborn': '>=0.12.0',
        'scipy': '>=1.10.0',
        'keras': '>=3.0.0',
        'transformers': '>=4.20.0',
        'opencv-python': '>=4.8.0',
        'cv2': '>=4.8.0',
    }

    def __init__(self):
        self.updated_deps = []
        self.detected_imports = set()

    def scan_project_imports(self, repo_path: str) -> Set[str]:
        """Scan all Python files to detect actual imports used"""
        all_imports = set()
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    imports = self._extract_imports_from_file(file_path)
                    all_imports.update(imports)
        print(f"📦 Detected imports: {sorted(all_imports)}")
        self.detected_imports = all_imports
        return all_imports

    def _extract_imports_from_file(self, file_path: str) -> Set[str]:
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split('.')[0])
        except (SyntaxError, UnicodeDecodeError, OSError):
            pass
        return imports

    def update_requirements_txt(self, repo_path: str) -> bool:
        """Update or create requirements.txt with detected dependencies"""
        req_path = os.path.join(repo_path, 'requirements.txt')
        self.scan_project_imports(repo_path)
        seen_packages = set()
        updated_lines = []

        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    updated_lines.append(line + '\n')
                    continue
                pkg_match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if not pkg_match:
                    updated_lines.append(line + '\n')
                    continue
                pkg_name = pkg_match.group(1).lower()
                seen_packages.add(pkg_name)

                if pkg_name in self.ML_DEPENDENCIES:
                    new_version = self.ML_DEPENDENCIES[pkg_name]
                    new_line = f"{pkg_name}{new_version}"
                    updated_lines.append(new_line + '\n')
                    self.updated_deps.append(f"{line} → {new_line}")
                else:
                    updated_lines.append(line + '\n')
        else:
            updated_lines.append("# Auto-generated requirements.txt\n")

        # Add missing detected deps
        for imp in self.detected_imports:
            imp = imp.lower()
            if imp in self.ML_DEPENDENCIES and imp not in seen_packages:
                new_line = f"{imp}{self.ML_DEPENDENCIES[imp]}"
                updated_lines.append(new_line + '\n')
                self.updated_deps.append(f"Added: {new_line}")

        with open(req_path, 'w') as f:
            f.writelines(updated_lines)

        print(f"✅ requirements.txt updated with {len(self.updated_deps)} changes")
        return True


    def update_setup_py(self, repo_path: str) -> bool:
        """Update setup.py with latest ML dependency versions"""
        setup_path = os.path.join(repo_path, 'setup.py')
        if not os.path.exists(setup_path):
            print("⚠️ No setup.py found, skipping setup update.")
            return False

        with open(setup_path, 'r') as f:
            content = f.read()

        updated_content = content
        for dep, version in self.ML_DEPENDENCIES.items():
            pattern = rf'(["\']){dep}[>=<!=]*[^"\',]*(["\'])'
            replacement = rf'\1{dep}{version}\2'
            new_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            if new_content != updated_content:
                self.updated_deps.append(f"Updated {dep} in setup.py → {version}")
                updated_content = new_content

        if updated_content != content:
            with open(setup_path, 'w') as f:
                f.write(updated_content)
            print("✅ setup.py dependencies upgraded successfully")
            return True

        print("ℹ️ setup.py already up-to-date.")
        return False
