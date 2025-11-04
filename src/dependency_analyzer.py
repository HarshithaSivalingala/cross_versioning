import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union


class DependencyAnalyzer:
    """
    Analyzes Python import dependencies to determine upgrade order.
    
    This helps upgrade files in the right order so that when we upgrade a file,
    we already know about the files it depends on.
    """
    
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.dependency_graph: Dict[str, List[str]] = {}
    
    def analyze_repository(self) -> Dict[str, List[str]]:
        """
        Build a complete dependency graph for all Python files in the repo.
        
        Returns:
            Dict mapping each file path to list of files it depends on
        """
        print("🔍 Analyzing repository dependencies...")
        
        python_files = self._find_python_files()
        
        for file_path in python_files:
            dependencies = self._extract_file_dependencies(file_path)
            self.dependency_graph[file_path] = dependencies
        
        print(f"✅ Analyzed {len(python_files)} files")
        return self.dependency_graph
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in the repository"""
        python_files = []
        
        for root, dirs, files in os.walk(self.repo_root):
            # Skip common directories we don't want to analyze
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', '.venv', 'venv', 
                '.tox', '.mypy_cache', 'node_modules', '.ml-upgrader',
                'build', 'dist', '*.egg-info'
            }]
            
            for filename in files:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    python_files.append(file_path)
        
        return python_files
    
    def _extract_file_dependencies(self, file_path: str) -> List[str]:
        """
        Extract local file dependencies from a Python file.
        
        Only returns dependencies on other files in the same repo,
        not external libraries like numpy or tensorflow.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=file_path)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            # If file has syntax errors or encoding issues, skip it
            print(f"⚠️  Could not parse {file_path}: {e}")
            return []
        
        import_info = self._parse_imports(tree)
        local_dependencies = []
        
        for import_type, *import_data in import_info:
            if import_type == 'absolute':
                module_name = import_data[0]
                dep_file = self._resolve_absolute_import(module_name)
                if dep_file:
                    local_dependencies.append(dep_file)
            
            elif import_type == 'relative':
                level, module_name = import_data
                dep_file = self._resolve_relative_import(level, module_name, file_path)
                if dep_file:
                    local_dependencies.append(dep_file)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in local_dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        
        return unique_deps
    
    def _parse_imports(self, tree: ast.AST) -> List[Tuple[str, ...]]:
        """
        Parse all import statements from AST.
        
        Returns list of tuples:
            ('absolute', 'module.name') for absolute imports
            ('relative', level, 'module.name') for relative imports
        """
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import foo, import foo.bar
                for alias in node.names:
                    imports.append(('absolute', alias.name))
            
            elif isinstance(node, ast.ImportFrom):
                # Handle: from foo import bar, from foo.bar import baz
                if node.level == 0:
                    # Absolute import: from foo.bar import baz
                    if node.module:
                        imports.append(('absolute', node.module))
                else:
                    # Relative import: from . import foo, from ..bar import baz
                    module_name = node.module or ''
                    imports.append(('relative', node.level, module_name))
        
        return imports
    
    def _resolve_absolute_import(self, module_name: str) -> Optional[str]:
        """
        Convert an absolute import to a file path.
        
        Examples:
            "core.utils" -> "/repo/core/utils.py"
            "src.models.yolo" -> "/repo/src/models/yolo.py"
        """
        # Skip standard library and external packages
        if self._is_external_module(module_name):
            return None
        
        # Convert module name to path: "core.utils" -> "core/utils"
        parts = module_name.split('.')
        
        # Try direct file path: core/utils.py
        file_path = os.path.join(self.repo_root, *parts) + '.py'
        if os.path.isfile(file_path):
            return os.path.abspath(file_path)
        
        # Try package __init__.py: core/utils/__init__.py
        init_path = os.path.join(self.repo_root, *parts, '__init__.py')
        if os.path.isfile(init_path):
            return os.path.abspath(init_path)
        
        return None
    
    def _is_external_module(self, module_name: str) -> bool:
        """Check if this is an external library (not local code)"""
        # Common external packages we want to ignore
        external_prefixes = [
            'numpy', 'np', 'tensorflow', 'tf', 'torch', 'pandas', 'pd',
            'sklearn', 'matplotlib', 'cv2', 'PIL', 'requests', 'flask',
            'django', 'pytest', 'unittest', 'os', 'sys', 'json',
            'collections', 're', 'math', 'random', 'datetime', 'time',
            'pathlib', 'typing', 'itertools', 'functools', 'operator',
            'pickle', 'csv', 'logging', 'argparse', 'subprocess',
            'threading', 'multiprocessing', 'queue', 'socket', 'http',
            'urllib', 'hashlib', 'base64', 'io', 'tempfile', 'shutil',
            'glob', 'zipfile', 'tarfile', 'gzip', 'warnings', 'traceback',
            'inspect', 'ast', 'dis', 'copy', 'dataclasses', 'enum',
            'abc', 'contextlib', 'weakref', 'gc', 'importlib'
        ]
        
        first_part = module_name.split('.')[0]
        return first_part in external_prefixes
    
    def _resolve_relative_import(
        self, 
        level: int, 
        module_name: str, 
        importing_file: str
    ) -> Optional[str]:
        """
        Handle relative imports like 'from . import foo' or 'from ..bar import baz'.
        
        Args:
            level: Number of dots (1 = current package, 2 = parent package, etc.)
            module_name: The module being imported (empty string for 'from . import foo')
            importing_file: The file doing the importing
        
        Examples:
            level=1, module="utils", file="/repo/core/models/yolo.py"
              → /repo/core/models/utils.py
            
            level=2, module="common", file="/repo/core/models/yolo.py"
              → /repo/core/common.py
            
            level=1, module="", file="/repo/core/models/yolo.py"
              → /repo/core/models/__init__.py
        """
        # Get the directory of the importing file
        current_dir = os.path.dirname(os.path.abspath(importing_file))
        
        # Navigate up 'level' directories
        # level=1 means current package (same directory)
        # level=2 means parent package (one directory up)
        target_dir = current_dir
        for _ in range(level - 1):
            parent = os.path.dirname(target_dir)
            
            # Safety check: don't go above repo root
            if not parent.startswith(self.repo_root):
                return None
            
            target_dir = parent
        
        # Now target_dir is where we should look for the module
        
        if not module_name:
            # "from . import something" or "from .. import something"
            # This imports from the package's __init__.py
            init_path = os.path.join(target_dir, '__init__.py')
            if os.path.isfile(init_path):
                return os.path.abspath(init_path)
            return None
        
        # "from .utils import foo" or "from ..common import bar"
        # Convert module path to file path
        parts = module_name.split('.')
        
        # Try direct file: .../utils.py
        file_path = os.path.join(target_dir, *parts) + '.py'
        if os.path.isfile(file_path):
            return os.path.abspath(file_path)
        
        # Try package __init__.py: .../utils/__init__.py
        init_path = os.path.join(target_dir, *parts, '__init__.py')
        if os.path.isfile(init_path):
            return os.path.abspath(init_path)
        
        return None
    
    def get_upgrade_order(self) -> List[str]:
        """
        Get the order in which files should be upgraded.
        
        Files with no dependencies come first, then files that depend on them, etc.
        This is called "topological sorting".
        
        Returns:
            List of file paths in upgrade order (dependencies before dependents)
        """
        if not self.dependency_graph:
            self.analyze_repository()
        
        return self._topological_sort()
    
    def _topological_sort(self) -> List[str]:
        """
        Sort files by dependencies using Kahn's algorithm.
        
        This ensures we upgrade dependencies before the files that use them.
        The algorithm processes files with no dependencies first, then gradually
        processes files as their dependencies are satisfied.
        """
        # Create a working copy of the graph
        graph = {node: list(deps) for node, deps in self.dependency_graph.items()}
        
        # Result list
        sorted_files = []
        
        # Find all nodes with no dependencies (leaves)
        no_deps = [node for node, deps in graph.items() if not deps]
        
        while no_deps:
            # Sort for consistency (alphabetical order)
            no_deps.sort()
            
            # Take a node with no dependencies
            node = no_deps.pop(0)
            sorted_files.append(node)
            
            # Find all nodes that depend on this node
            # and remove this dependency from them
            for other_node in graph:
                if node in graph[other_node]:
                    graph[other_node].remove(node)
                    
                    # If this node now has no dependencies, add it to the queue
                    if not graph[other_node]:
                        no_deps.append(other_node)
        
        # Check for circular dependencies
        remaining = [node for node, deps in graph.items() if deps]
        if remaining:
            print(f"⚠️  Found {len(remaining)} files with circular dependencies:")
            for node in remaining[:5]:  # Show first 5
                deps = graph[node]
                print(f"   {os.path.basename(node)} → {[os.path.basename(d) for d in deps[:3]]}")
            if len(remaining) > 5:
                print(f"   ... and {len(remaining) - 5} more")
            
            # Add them at the end in alphabetical order
            sorted_files.extend(sorted(remaining))
        
        return sorted_files
    
    def get_direct_dependencies(self, file_path: str) -> List[str]:
        """Get the list of files that this file directly imports from"""
        return self.dependency_graph.get(file_path, [])
    
    def print_dependency_tree(self, max_depth: int = 3) -> None:
        """
        Print a visual representation of the dependency tree.
        Useful for debugging and understanding the project structure.
        """
        if not self.dependency_graph:
            self.analyze_repository()
        
        print("\n📊 Dependency Tree:")
        print("=" * 60)
        
        # Find root files (files with no dependencies)
        roots = [f for f, deps in self.dependency_graph.items() if not deps]
        
        if not roots:
            print("⚠️  No root files found (all files have dependencies)")
            # Show files with fewest dependencies
            sorted_by_deps = sorted(
                self.dependency_graph.items(),
                key=lambda x: len(x[1])
            )
            roots = [sorted_by_deps[0][0]] if sorted_by_deps else []
        
        visited = set()
        
        def print_tree(file_path: str, depth: int = 0, prefix: str = ""):
            if depth > max_depth or file_path in visited:
                return
            
            visited.add(file_path)
            rel_path = os.path.relpath(file_path, self.repo_root)
            
            print(f"{prefix}{rel_path}")
            
            # Find files that depend on this one
            dependents = [
                f for f, deps in self.dependency_graph.items()
                if file_path in deps
            ]
            
            for i, dependent in enumerate(sorted(dependents)):
                is_last = i == len(dependents) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                branch = "└── " if is_last else "├── "
                print_tree(dependent, depth + 1, prefix + branch)
        
        for root in sorted(roots)[:10]:  # Show first 10 roots
            print_tree(root)
            print()
        
        if len(roots) > 10:
            print(f"... and {len(roots) - 10} more root files")
        
        print("=" * 60)


def extract_interface_summary(file_path: str) -> str:
    """
    Extract just the function signatures and class definitions from a file.
    
    This gives us the "interface" without all the implementation details.
    We use this to tell the LLM about dependencies without bloating the prompt.
    
    Returns:
        A compact summary like:
        "def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold) -> np.ndarray
         class YOLOv3:
             def __init__(self, input_size, num_classes)
             def predict(self, image) -> dict"
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return ""
    
    summary_lines = []
    
    # Walk only top-level nodes (not nested functions/classes)
    for node in tree.body:
        # Extract function definitions
        if isinstance(node, ast.FunctionDef):
            # Skip private functions (those starting with _)
            if node.name.startswith('_') and not node.name.startswith('__'):
                continue
            
            # Build function signature
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                # Include type annotation if available
                if arg.annotation:
                    try:
                        arg_name += f": {ast.unparse(arg.annotation)}"
                    except:
                        pass  # Skip if unparsing fails
                args.append(arg_name)
            
            signature = f"def {node.name}({', '.join(args)})"
            
            # Add return type if available
            if node.returns:
                try:
                    signature += f" -> {ast.unparse(node.returns)}"
                except:
                    pass  # Skip if unparsing fails
            
            summary_lines.append(signature)
        
        # Extract class definitions
        elif isinstance(node, ast.ClassDef):
            summary_lines.append(f"\nclass {node.name}:")
            
            # Extract public methods (only direct methods, not nested)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name.startswith('_') and not item.name.startswith('__'):
                        continue
                    
                    # Build method signature
                    args = []
                    for arg in item.args.args:
                        arg_name = arg.arg
                        if arg.annotation:
                            try:
                                arg_name += f": {ast.unparse(arg.annotation)}"
                            except:
                                pass
                        args.append(arg_name)
                    
                    method_sig = f"def {item.name}({', '.join(args)})"
                    
                    # Add return type
                    if item.returns:
                        try:
                            method_sig += f" -> {ast.unparse(item.returns)}"
                        except:
                            pass
                    
                    summary_lines.append(f"    {method_sig}")
    
    return '\n'.join(summary_lines)


def visualize_dependencies(repo_path: str, output_file: Optional[str] = None) -> None:
    """
    Create a visual representation of dependencies.
    Optionally save to a file.
    
    Args:
        repo_path: Path to repository
        output_file: Optional path to save visualization (e.g., "deps.txt")
    """
    analyzer = DependencyAnalyzer(repo_path)
    analyzer.analyze_repository()
    
    upgrade_order = analyzer.get_upgrade_order()
    
    output = []
    output.append("=" * 70)
    output.append("DEPENDENCY ANALYSIS REPORT")
    output.append("=" * 70)
    output.append(f"\nRepository: {repo_path}")
    output.append(f"Total Python files: {len(analyzer.dependency_graph)}")
    output.append("")
    
    # Show upgrade order
    output.append("UPGRADE ORDER (dependencies first):")
    output.append("-" * 70)
    for i, file_path in enumerate(upgrade_order, 1):
        rel_path = os.path.relpath(file_path, repo_path)
        deps = analyzer.get_direct_dependencies(file_path)
        dep_count = len(deps)
        output.append(f"{i:3d}. {rel_path:50s} ({dep_count} deps)")
    
    output.append("")
    output.append("DEPENDENCY DETAILS:")
    output.append("-" * 70)
    
    for file_path in sorted(analyzer.dependency_graph.keys()):
        rel_path = os.path.relpath(file_path, repo_path)
        deps = analyzer.get_direct_dependencies(file_path)
        
        if deps:
            output.append(f"\n{rel_path}:")
            for dep in deps:
                dep_rel = os.path.relpath(dep, repo_path)
                output.append(f"  → {dep_rel}")
        else:
            output.append(f"\n{rel_path}: (no dependencies)")
    
    output.append("")
    output.append("=" * 70)
    
    result = "\n".join(output)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"✅ Dependency report saved to {output_file}")
    else:
        print(result)


# Convenience function for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dependency_analyzer.py <repo_path> [output_file]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_dependencies(repo_path, output_file)