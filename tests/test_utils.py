import pytest
import tempfile
import os
from src.utils import (
    read_file,
    write_file,
    build_prompt,
    extract_api_changes,
    generate_diff,
    detect_tf1_usage,
)

class TestUtils:
    
    def test_read_write_file(self):
        """Test file read/write operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test", "file.py")
            content = "print('Hello, World!')"
            
            # Test write
            write_file(file_path, content)
            assert os.path.exists(file_path)
            
            # Test read
            read_content = read_file(file_path)
            assert read_content == content
    
    def test_build_prompt_without_error(self):
        """Test prompt building without error context"""
        code = "import tensorflow as tf\nprint(tf.__version__)"
        prompt = build_prompt(code)
        
        assert "migration assistant" in prompt.lower()
        assert "tensorflow" in prompt
        assert code in prompt
        assert "failed with error" not in prompt
    
    def test_build_prompt_with_error(self):
        """Test prompt building with error context"""
        code = "import tensorflow as tf\nprint(tf.__version__)"
        error = "ModuleNotFoundError: No module named 'tensorflow'"
        prompt = build_prompt(code, error)
        
        assert "failed with this error" in prompt
        assert error in prompt
        assert code in prompt
    
    def test_extract_api_changes(self):
        """Test API change detection"""
        old_code = """
import tensorflow as tf
sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.layers.dense(x, 10)
result = sess.run(y)
"""
        
        new_code = """
import tensorflow as tf
# Eager execution, no sessions needed
x = tf.Variable(tf.zeros([None, 784]))
y = tf.keras.layers.Dense(10)(x)
result = y.numpy()
"""
        
        changes = extract_api_changes(old_code, new_code)
        
        # Should detect TF 1.x → 2.x changes
        tf_session_removed = any("tf.Session" in change for change in changes)
        tf_layers_migrated = any("tf.layers" in change for change in changes)
        
        assert tf_session_removed or tf_layers_migrated
    
    def test_generate_diff(self):
        """Test diff generation"""
        old_content = "import numpy as np\nx = np.asscalar(arr[0])"
        new_content = "import numpy as np\nx = arr[0].item()"
        filename = "test.py"
        
        diff = generate_diff(old_content, new_content, filename)
        
        assert "old/test.py" in diff
        assert "new/test.py" in diff
        assert "asscalar" in diff
        assert "item()" in diff

    def test_detect_tf1_usage_flags_patterns(self):
        code = """
import tensorflow as tf
with tf.compat.v1.Session() as sess:
    x = tf.placeholder(tf.float32)
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={x: [1.0]})
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "tf.compat.v1" in message
        assert "feed_dict" in message

    def test_detect_tf1_usage_returns_none_for_non_tf_code(self):
        code = "import torch\nprint(torch.__version__)"

        message = detect_tf1_usage(code)

        assert message is None

    def test_detect_tf1_usage_flags_compat_mode(self):
        code = """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x = tf.placeholder(tf.float32)
with tf.Session() as sess:
    print(sess.run(x, feed_dict={x: 1.0}))
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "compat.v1" in message
        assert "disable" in message

    def test_detect_tf1_usage_flags_disable_without_alias(self):
        code = """
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as sess:
    print(sess.run(tf.constant(1)))
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "tf.compat.v1" in message

    def test_detect_tf1_usage_flags_estimators(self):
        code = """
import tensorflow as tf
classifier = tf.estimator.DNNClassifier(hidden_units=[10], feature_columns=[])
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "tf.estimator" in message

    def test_detect_tf1_usage_flags_input_without_model(self):
        code = """
import tensorflow as tf

x = tf.keras.Input(shape=(1,))
y = x * 2.0
print(y)
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "tf.keras.Input" in message or "tf.keras.Model" in message

    def test_detect_tf1_usage_flags_backend_session_calls(self):
        code = """
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
tf.keras.backend.set_session(tf.Session())
"""

        message = detect_tf1_usage(code)

        assert message is not None
        assert "backend" in message
