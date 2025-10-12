import pytest
import tempfile
import os
from src.validator import validate_syntax, validate_code

class TestValidator:
    
    def test_validate_syntax_valid(self):
        """Test syntax validation with valid Python code"""
        code = "import os\nprint('Hello, World!')"
        is_valid, error = validate_syntax(code)
        
        assert is_valid == True
        assert error is None
    
    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid Python code"""
        code = "import os\nprint('Hello, World!"  # Missing closing quote
        is_valid, error = validate_syntax(code)
        
        assert is_valid == False
        assert error is not None
        assert "syntax error" in error.lower()
    
    def test_validate_code_valid(self):
        """Test full code validation with valid file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import sys\nprint('Hello, World!')")
            f.flush()
            
            is_valid, error = validate_code(f.name)
            
            # Clean up
            os.unlink(f.name)
            
            assert is_valid == True
            assert error is None
    
    def test_validate_code_syntax_error(self):
        """Test full code validation with syntax error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import sys\nprint('Hello, World!")  # Missing quote
            f.flush()
            
            is_valid, error = validate_code(f.name)
            
            # Clean up
            os.unlink(f.name)
            
            assert is_valid == False
            assert error is not None