import tempfile
import os
from unittest.mock import patch
from src.repo_upgrader import upgrade_repo

class TestIntegration:

    @patch('src.dependency_upgrader.PyPIVersionResolver.requires_dist')
    @patch('src.dependency_upgrader.PyPIVersionResolver.latest_version')
    @patch('src.llm_interface.call_llm')
    def test_full_repo_upgrade(self, mock_llm, mock_latest_version, mock_requires_dist):
        """Test full repository upgrade workflow"""

        # Mock LLM to return upgraded code
        def mock_llm_response(prompt):
            if "tf.Session" in prompt:
                return """
import tensorflow as tf

# Upgraded to TF 2.x eager execution
x = tf.Variable(tf.zeros([1, 784]))
y = tf.keras.layers.Dense(10)(x)
result = y.numpy()
"""
            return "# Upgraded code\nprint('Hello, World!')"
        
        mock_llm.side_effect = mock_llm_response
        version_map = {
            "tensorflow": "2.15.0",
            "numpy": "1.24.0",
        }
        mock_latest_version.side_effect = lambda pkg: version_map.get((pkg or "").lower())
        requires_map = {
            ("tensorflow", "2.15.0"): ["numpy>=1.24.0"],
        }
        mock_requires_dist.side_effect = lambda pkg, ver: requires_map.get(((pkg or "").lower(), ver), [])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample old repository
            old_repo = os.path.join(temp_dir, "old_repo")
            new_repo = os.path.join(temp_dir, "new_repo")
            
            os.makedirs(old_repo)
            venv_dir = os.path.join(old_repo, ".venv")
            os.makedirs(venv_dir)
            with open(os.path.join(venv_dir, "pyvenv.cfg"), "w") as handle:
                handle.write("home = /usr/bin/python\n")
            
            # Create sample Python files
            with open(os.path.join(old_repo, "model.py"), 'w') as f:
                f.write("""
import tensorflow as tf

sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.layers.dense(x, 10)
result = sess.run(y)
""")
            
            with open(os.path.join(old_repo, "requirements.txt"), 'w') as f:
                f.write("tensorflow==1.15.0\nnumpy==1.18.0\n")
            
            # Set mock environment
            os.environ["OPENROUTER_API_KEY"] = "test-key"
            
            # Run upgrade
            report_path = upgrade_repo(old_repo, new_repo)
            
            # Verify results
            assert os.path.exists(new_repo)
            assert os.path.exists(report_path)
            assert not os.path.exists(os.path.join(new_repo, ".venv"))
            
            # Check if files were processed
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            assert "ML Repository Upgrade Report" in report_content
            assert "model.py" in report_content
