## Sample Legacy ML Project

This project bundles several legacy code paths so you can try out the ML
upgrader end-to-end without touching your own repositories.

### What's Included

- **TensorFlow 1.x graph** (`legacy/tf_model.py`) using `tf.Session`,
  `tf.placeholder`, and `tf.layers`.
- **PyTorch legacy training loop** (`legacy/torch_model.py`) based on
  `torch.autograd.Variable` and functional activations.
- **NumPy deprecations** (`utils/data_utils.py`) such as `np.asscalar` and the
  `np.float` alias.
- **Synthetic data pipeline** with setup commands that generate and preprocess
  a CSV dataset before the runtime smoke test runs.

### Runtime Validation

The `ml_upgrader_runtime.json` file enables the following behaviour:

1. `["python", "scripts/download_data.py"]` – generates a synthetic dataset.
2. `"python scripts/preprocess_data.py"` – converts the CSV into a NumPy `.npy`
   bundle and writes metadata.
3. `["python", "main.py"]` – loads the processed data, prints summary stats, and
   records them to disk. If the dataset is missing, the script now runs the two
   setup commands automatically as a fallback. Set `ENABLE_LEGACY_TRAINING=1`
   to execute the TensorFlow and PyTorch examples after upgrading.

The runtime config sets `skip_install` to `true` so these quick smoke tests do
not attempt to install heavy frameworks. You can flip this to `false` after the
upgrader modernises the dependencies if you want to exercise the full stack.
