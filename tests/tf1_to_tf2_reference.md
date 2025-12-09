# TensorFlow 1.x → 2.x Upgrade Reference

Use this checklist when manually reviewing upgraded files (or the live LLM
results) to confirm they follow TensorFlow 2 best practices.

| TF1 Pattern / API | TF2 Expectation | Why / Notes |
| --- | --- | --- |
| `tf.Session`, `with tf.Session()` | Remove entirely; rely on eager execution (`tensor.numpy()`), or wrap code in `@tf.function` if you need graph mode | TF2 executes ops eagerly; sessions reintroduce TF1 semantics |
| `tf.compat.v1.*`, `tf.compat.v1.Session`, `tf.compat.v1.disable_*` | Never use unless explicitly whitelisting legacy graph mode (which we do not); replace code with native TF2 equivalents | Compat APIs keep graph execution, defeating the migration |
| `tf.placeholder(...)` + `feed_dict=` | Replace with eager tensors (`tf.constant`, `tf.Variable`) or `tf.keras.Input` only when immediately constructing a `tf.keras.Model` | Feed dicts require sessions; TF2 inputs should be concrete tensors |
| `tf.global_variables_initializer()` / `tf.compat.v1.global_variables_initializer()` | Remove; variables initialize when first used in eager mode or through `model.build` | Explicit initializers are redundant |
| `tf.get_variable`, `tf.variable_scope`, manual weight sharing | Use `tf.Variable` or `tf.keras.layers`/`tf.keras.Model` (e.g., a shared `tf.keras.layers.Dense`) | Variable scopes are TF1 graph constructs |
| `tf.layers.*` | Swap to `tf.keras.layers.*` and build a functional or sequential model | `tf.layers` is deprecated; `tf.keras` is the supported high-level API |
| Manual training loop with `tf.train.Optimizer.minimize` | Use `tf.keras.optimizers.*` with `tf.GradientTape`, or `model.fit` when training full models | Gradient tape mirrors eager autograd |
| `tf.estimator.*` models | Convert to `tf.keras` models with `compile`/`fit` and NumPy or `tf.data` inputs | Estimators are frozen TF1 APIs |
| `tf.summary.FileWriter` | Replace with `tf.summary.create_file_writer` context blocks | TF2 summary writer integrates with eager execution |
| `tf.saved_model.builder.SavedModelBuilder` | Use `tf.keras.Model.save(...)` or `tf.saved_model.save(model, path)` | Builder API is TF1-only |
| `tf.data` one-shot iterators (`make_one_shot_iterator`, `sess.run(next_elem)`) | Iterate directly (`for elem in dataset:`) and call `.numpy()` | Datasets are native Python iterables in TF2 |
| `tf.nn.rnn_cell`, `tf.nn.dynamic_rnn` | Replace with `tf.keras.layers.RNN`/`LSTM` and build a `tf.keras.Model` | High-level Keras RNN layers cover these features |
| `tf.contrib.*` | Replace with the nearest `tf.keras` / core TF2 equivalent | `tf.contrib` was removed |
| `tf.keras.backend.set_session/get_session/set_learning_phase` | Remove; these functions manage TF1 graph sessions and phases | TF2 manages eager/graph state automatically |
| Control-flow ops (`tf.while_loop`, `tf.cond`) on scalars with sessions | Prefer native Python control flow or wrap in `tf.function` if necessary | Eager mode executes Python loops directly |

## Review Flow

1. **Search for banned APIs** – look for any occurrence of `tf.Session`,
   `tf.compat.v1`, `feed_dict`, `tf.placeholder`, or backend session helpers.
2. **Confirm inputs/outputs are eager** – tensors should be created with
   `tf.constant`/`tf.Variable` or via `tf.keras.Input` that immediately feeds a
   `tf.keras.Model`. Printing results should rely on `.numpy()`.
3. **Prefer high-level APIs** – training/tracking should use `tf.keras` models,
   optimizers, and `model.fit`/`GradientTape` instead of manual graph code.
4. **Check persistence APIs** – saving, summaries, dataset iteration, and RNNs
   should use the TF2-native calls listed above.

Keeping this sheet next to `tests/llm_live_results_tensorflow.txt` makes it easy
to verify that each upgraded snippet satisfies the expected TF2 idioms.
