import pytest
import tempfile
import os
from src.dependency_upgrader import DependencyUpdater

class TestDependencyUpdater:
    
    def test_update_requirements_txt(self):
        """Test requirements.txt updating"""
        updater = DependencyUpdater()
        
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
            
            assert "tensorflow>=2.15.0" in content
            assert "numpy>=1.24.0" in content
            assert "requests==2.25.0" in content  # Non-ML library unchanged
    
    def test_update_nonexistent_requirements(self):
        """Test handling of non-existent requirements.txt"""
        updater = DependencyUpdater()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = updater.update_requirements_txt(temp_dir)
            assert success == False