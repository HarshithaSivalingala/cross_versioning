import os
import re
from typing import Dict, List, Optional

class DependencyUpdater:
    """Update ML dependencies to latest compatible versions"""
    
    ML_DEPENDENCIES = {
        'tensorflow': '>=2.15.0',
        'torch': '>=2.0.0',
        'numpy': '>=1.24.0',
        'jax': '>=0.4.0',
        'jaxlib': '>=0.4.0',
        'scikit-learn': '>=1.3.0',
        'pandas': '>=2.0.0',
        'matplotlib': '>=3.7.0',
        'seaborn': '>=0.12.0',
        'scipy': '>=1.10.0',
        'keras': '>=3.0.0',
        'transformers': '>=4.20.0',
        'opencv-python': '>=4.8.0',
    }
    
    def __init__(self):
        self.updated_deps = []
    
    def update_requirements_txt(self, repo_path: str) -> bool:
        """Update requirements.txt with latest ML dependency versions"""
        req_path = os.path.join(repo_path, 'requirements.txt')
        if not os.path.exists(req_path):
            return False
            
        with open(req_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                updated_lines.append(line + '\n')
                continue
                
            # Parse dependency line
            pkg_match = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if pkg_match:
                pkg_name = pkg_match.group(1).lower()
                if pkg_name in self.ML_DEPENDENCIES:
                    new_version = self.ML_DEPENDENCIES[pkg_name]
                    old_line = line
                    new_line = f"{pkg_name}{new_version}"
                    updated_lines.append(new_line + '\n')
                    self.updated_deps.append(f"{old_line} → {new_line}")
                    continue
            
            updated_lines.append(line + '\n')
        
        with open(req_path, 'w') as f:
            f.writelines(updated_lines)
        
        return True
    
    def update_setup_py(self, repo_path: str) -> bool:
        """Update setup.py dependencies"""
        setup_path = os.path.join(repo_path, 'setup.py')
        if not os.path.exists(setup_path):
            return False
            
        with open(setup_path, 'r') as f:
            content = f.read()
        
        # Look for install_requires or requirements patterns
        updated_content = content
        for dep, version in self.ML_DEPENDENCIES.items():
            # Match patterns like 'tensorflow>=1.0' or "tensorflow==1.15"
            pattern = rf'(["\']){dep}[>=<!=]*[^"\',]*(["\'])'
            replacement = rf'\1{dep}{version}\2'
            new_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            
            if new_content != updated_content:
                self.updated_deps.append(f"Updated {dep} in setup.py")
                updated_content = new_content
        
        if updated_content != content:
            with open(setup_path, 'w') as f:
                f.write(updated_content)
            return True
        
        return False
