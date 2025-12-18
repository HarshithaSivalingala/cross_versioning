import os
import re
import shutil
import difflib
import subprocess
import tempfile
from typing import Dict, List, Optional

_UNNECESSARY_DIRECTORIES = {"__MACOSX", "__pycache__"}
_UNNECESSARY_FILES = {".DS_Store", "Thumbs.db"}

TF_COMPREHENSIVE_GUIDE = """
=== COMPREHENSIVE TENSORFLOW 1.x → 2.x MIGRATION GUIDE ===

CORE EXECUTION MODEL:
TF1: Build graph → Create session → Run operations
TF2: Eager execution (operations run immediately)

CRITICAL REMOVED APIS (DO NOT USE):
- tf.Session / tf.InteractiveSession
- tf.placeholder
- tf.get_variable / tf.variable_scope
- tf.contrib.* (entire module removed)
- slim.* (use tf.data and tf.keras instead)
- tf.flags (use argparse or absl.flags)
- tf.app.run
- tf.logging (use Python logging)
- tf.layers.* (use tf.keras.layers)
- Queue-based data loading

MIGRATION PATTERNS:

1. SESSIONS → EAGER EXECUTION:
Before: sess = tf.Session(); result = sess.run(op, feed_dict={x: data})
After:  result = op(data)  # Direct execution, no session needed

2. PLACEHOLDERS → FUNCTION PARAMETERS:
Before: x = tf.placeholder(tf.float32, [None, 784])
After:  def model(x):  # x is now a regular function parameter

3. VARIABLES:
Before: v = tf.get_variable("v", shape=[1])
After:  v = tf.Variable([0.0], name="v")

4. LAYERS:
Before: tf.layers.dense(x, 128)
After:  tf.keras.layers.Dense(128)(x)

5. SLIM/DATASET → TF.DATA:
Before:
  reader = tf.TFRecordReader
  decoder = slim.tfexample_decoder.TFExampleDecoder(keys, handlers)
  return slim.dataset.Dataset(decoder=decoder, ...)

After:
  # CRITICAL: Resolve file patterns first!
  file_pattern = os.path.join(dir, pattern % split)
  filenames = tf.io.gfile.glob(file_pattern)  # Must resolve pattern to actual files
  
  dataset = tf.data.TFRecordDataset(filenames)
  
  def parse_fn(example):
      feature_spec = {
          'image/encoded': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
      }
      parsed = tf.io.parse_single_example(example, feature_spec)
      image = tf.io.decode_image(parsed['image/encoded'], channels=3)
      return image, parsed['label']
  
  dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset  # Return tf.data.Dataset object, NOT dict

6. TRAINING LOOPS:
Before:
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for step in range(1000):
          _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})

After:
  @tf.function
  def train_step(x, y):
      with tf.GradientTape() as tape:
          predictions = model(x, training=True)
          loss = loss_fn(y, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return loss
  
  for x_batch, y_batch in dataset:
      loss = train_step(x_batch, y_batch)

COMMON MISTAKES TO AVOID:

❌ dataset = tf.data.TFRecordDataset(filenames=file_pattern)
   # file_pattern is a string like "data_%s.tfr", not actual files!
✅ filenames = tf.io.gfile.glob(file_pattern)
   dataset = tf.data.TFRecordDataset(filenames)

❌ items_to_handlers = {'image': tf.io.decode_image(tf.io.FixedLenFeature(...))}
   # Feature specs are for parsing only, not decoding!
✅ parsed = tf.io.parse_single_example(example, feature_spec)
   image = tf.io.decode_image(parsed['image/encoded'])

❌ return {'data_sources': file_pattern, 'reader': reader, 'decoder': decoder}
   # Must return actual tf.data.Dataset, not metadata!
✅ return dataset

❌ dataset.map(parse_fn)
   # Missing performance optimization!
✅ dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

❌ default_value=tf.zeros([], dtype=tf.int64)
   # Unnecessarily complex!
✅ default_value=-1

❌ from __future__ import absolute_import
   # Python 2 compatibility not needed!
✅ # Remove these lines

❌ VarLenFeature without conversion
✅ sparse_tensor = parsed['var_len_feature']
   dense = tf.sparse.to_dense(sparse_tensor)

PERFORMANCE BEST PRACTICES:
- Always use num_parallel_calls=tf.data.AUTOTUNE in .map()
- Use @tf.function decorator for training loops
- Use tf.data.AUTOTUNE for prefetching
- Batch before map when possible

API RENAMES:
- tf.FixedLenFeature → tf.io.FixedLenFeature
- tf.VarLenFeature → tf.io.VarLenFeature
- tf.TFRecordReader → tf.data.TFRecordDataset
- tf.parse_single_example → tf.io.parse_single_example
- tf.decode_raw → tf.io.decode_raw
- tf.image.decode_jpeg → tf.io.decode_jpeg
"""

def read_file(path: str) -> str:
    """Read file content with encoding handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def write_file(path: str, content: str) -> None:
    """Write content to file with directory creation"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def prune_directory(root: str) -> None:
    """Remove artifacts such as __MACOSX folders and resource forks."""
    for current_root, dirnames, filenames in os.walk(root):
        for dirname in list(dirnames):
            if dirname in _UNNECESSARY_DIRECTORIES or dirname.startswith("._"):
                shutil.rmtree(os.path.join(current_root, dirname), ignore_errors=True)
                dirnames.remove(dirname)
        for filename in list(filenames):
            if (
                filename in _UNNECESSARY_FILES
                or filename.startswith("._")
                or filename.endswith(".pyc")
            ):
                try:
                    os.remove(os.path.join(current_root, filename))
                except OSError:
                    continue

def is_probably_binary(path: str, sample_size: int = 2048) -> bool:
    """Heuristic to detect binary files (null bytes or low text ratio)."""
    try:
        with open(path, "rb") as fh:
            sample = fh.read(sample_size)
    except OSError:
        return False

    if not sample:
        return False

    if b"\x00" in sample:
        return True

    text_chars = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
    text_set = set(text_chars)
    text_count = sum(1 for byte in sample if byte in text_set)
    ratio = text_count / len(sample)
    return ratio < 0.85

def should_skip_for_upgrade(path: str) -> Optional[str]:
    """Return reason string if file should not be upgraded."""
    parts = os.path.normpath(path).split(os.sep)
    if any(part == "__MACOSX" for part in parts):
        return "Skipped macOS resource fork metadata"

    filename = os.path.basename(path)
    if filename.startswith("._"):
        return "Skipped macOS resource fork file"

    if is_probably_binary(path):
        return "Skipped binary/non-text file"

    return None

def preprocess_with_official_tool(code: str) -> str:
    """Use TensorFlow's official tf_upgrade_v2 as first pass."""
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_input = f.name
        
        temp_output = temp_input.replace('.py', '_v2.py')
        
        # Run official converter
        result = subprocess.run([
            'tf_upgrade_v2',
            '--infile', temp_input,
            '--outfile', temp_output,
            '--reportfile', os.devnull
        ], capture_output=True, text=True, timeout=30)
        
        # Read upgraded code
        if os.path.exists(temp_output):
            with open(temp_output) as f:
                upgraded = f.read()
            
            # Cleanup
            os.unlink(temp_input)
            os.unlink(temp_output)
            
            # Only return if different from input
            if upgraded.strip() and upgraded != code:
                return upgraded
        
    except Exception as e:
        # If official tool fails, just continue with original code
        pass
    
    return code

def build_prompt(code: str, error: Optional[str] = None) -> str:
    """Build comprehensive prompt for LLM to upgrade ML code."""
    
    base_prompt = (
        "You are an expert Python ML code migration assistant.\n"
        "Upgrade the following Python code to be fully compatible with the latest stable version(s) "
        "of ONLY the libraries it already uses.\n\n"
        f"{TF_COMPREHENSIVE_GUIDE}\n\n"
        "⚠️ ADDITIONAL RULES:\n"
        "- Do NOT convert between frameworks (e.g., keep TensorFlow code in TensorFlow, PyTorch in PyTorch).\n"
        "- Do NOT add new frameworks unless already imported in the code.\n"
        "- Preserve all functionality and logic exactly.\n"
        "- Apply only necessary migrations (remove deprecated APIs, update function signatures, fix types).\n"
        "- Focus on correctness, performance, and idiomatic TF2 style.\n"
        "- Always return the ENTIRE corrected code.\n"
        "```python\n"
        "# upgraded code here\n"
        "```"
    )
    
    if error:
        return (
            f"{base_prompt}\n\n"
            "The previously upgraded code failed with this error:\n"
            f"{error}\n\n"
            "Please fix the issue and return the full corrected file.\n\n"
            f"Code:\n{code}\n"
        )
    else:
        return (
            f"{base_prompt}\n\n"
            "Code to upgrade:\n"
            f"{code}\n"
        )

def build_prompt_with_context(
    code: str, 
    dependency_context: Dict[str, str],
    error: Optional[str] = None
) -> str:
    """Build prompt for LLM with awareness of dependency interfaces."""
    
    base_prompt = (
        "You are an expert Python ML code migration assistant.\n"
        "Upgrade the following Python code to be fully compatible with the latest stable version(s) "
        "of ONLY the libraries it already uses.\n\n"
        f"{TF_COMPREHENSIVE_GUIDE}\n\n"
        "⚠️ ADDITIONAL RULES:\n"
        "- Do NOT convert between frameworks (e.g., keep TensorFlow code in TensorFlow, PyTorch in PyTorch).\n"
        "- Do NOT add new frameworks unless already imported in the code.\n"
        "- Preserve all functionality and logic exactly.\n"
        "- Apply only necessary migrations (remove deprecated APIs, update function signatures, fix types).\n"
        "- Focus on correctness, performance, and idiomatic TF2 style.\n"
        "- Always return the ENTIRE corrected code.\n"
    )
    
    # Add dependency context if available
    if dependency_context:
        context_section = "\n\n📚 DEPENDENCY CONTEXT:\n"
        context_section += "This file imports from other files in the repository. "
        context_section += "Here are their current interfaces (already upgraded):\n\n"
        
        for dep_path, interface in dependency_context.items():
            dep_name = os.path.basename(dep_path)
            context_section += f"### {dep_name} ###\n"
            if interface.strip():
                context_section += interface + "\n\n"
            else:
                context_section += "(empty or no public interface)\n\n"
        
        context_section += (
            "IMPORTANT: Your upgraded code MUST be compatible with these interfaces.\n"
            "- Match function signatures exactly\n"
            "- Use the same return types\n"
            "- Don't assume different APIs than shown above\n\n"
        )
        
        base_prompt += context_section
    
    base_prompt += "```python\n# upgraded code here\n```"
    
    # Add error context if this is a retry
    if error:
        return (
            f"{base_prompt}\n\n"
            "The previously upgraded code failed with this error:\n"
            f"{error}\n\n"
            "Please fix the issue and return the full corrected file.\n\n"
            f"Code to upgrade:\n{code}\n"
        )
    else:
        return (
            f"{base_prompt}\n\n"
            "Code to upgrade:\n"
            f"{code}\n"
        )

def extract_api_changes(old_code: str, new_code: str) -> List[str]:
    """Extract API changes between old and new code"""
    changes = []
    
    # Common API patterns to detect
    patterns = {
        r'tf\.Session\(\)': 'Removed tf.Session (TF 1.x → 2.x)',
        r'tf\.placeholder': 'Replaced tf.placeholder with function parameters',
        r'slim\.': 'Replaced slim.* with tf.data/tf.keras',
        r'tf\.contrib\.': 'Removed tf.contrib (deprecated in TF 2.x)',
        r'tf\.get_variable': 'Replaced tf.get_variable with tf.Variable',
        r'tf\.layers\.': 'Migrated tf.layers to tf.keras.layers',
        r'np\.asscalar': 'Replaced np.asscalar with .item()',
        r'np\.int\b': 'Replaced np.int with int',
        r'np\.float\b': 'Replaced np.float with float',
        r'torch\.cuda\.FloatTensor': 'Updated torch.cuda.FloatTensor to modern tensor creation',
        r'torch\.autograd\.Variable': 'Removed torch.autograd.Variable (no longer needed)',
    }
    
    for pattern, description in patterns.items():
        if re.search(pattern, old_code) and not re.search(pattern, new_code):
            changes.append(description)
    
    return changes

def generate_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate unified diff between old and new content"""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"old/{filename}",
        tofile=f"new/{filename}",
        n=3
    )
    return ''.join(diff)