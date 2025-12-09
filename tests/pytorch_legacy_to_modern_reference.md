# PyTorch Legacy → Modern Upgrade Reference

Use this sheet when checking upgraded PyTorch snippets (e.g., those written to
`tests/llm_live_results_pytorch.txt`). It summarizes the migrations encoded in
`tests/pytorch_conversion_dataset.json` so you can quickly confirm that the LLM
outputs match modern PyTorch idioms.

| PyTorch 0.x/1.x Pattern | Modern PyTorch Expectation | Why / Notes |
| --- | --- | --- |
| `torch.autograd.Variable(tensor, requires_grad=...)` | Use tensors directly (`tensor.requires_grad_()`) | `Variable` merged with `Tensor`; redundant wrapper |
| Legacy constructors `torch.FloatTensor(...)`, `torch.cuda.FloatTensor(...)`, `torch.Tensor(np_array)` | Prefer `torch.tensor(...)`/`torch.asarray` or `.to(device)` chaining | New tensor factory APIs respect dtype/device semantics |
| `.data` attribute for in-place updates (`param.data -= lr * grad`) | Use `with torch.no_grad(): param -= lr * grad` | `.data` bypasses autograd and is unsafe |
| `torch.nn.functional.softmax(logits)` without `dim` | Always pass the `dim` argument (`dim=-1`) | Default dimension is deprecated/ambiguous |
| `torch.load(..., map_location=lambda storage, loc: storage)` | Use `map_location=torch.device("cpu")` or the newer `torch.load(..., weights_only=True)` when applicable | Lambda form was deprecated |
| `optimizer = torch.optim.SGD(params, lr); optimizer.step(); optimizer.zero_grad()` mixed with manual `tensor.backward(torch.ones_like(tensor))` | Keep the standard loop: `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()` | Ensures grads accumulate correctly |
| `torch.nn.utils.rnn.pack_padded_sequence(..., enforce_sorted=False)` preceded by manual sorting | Prefer built-in `batch_first`/`enforce_sorted` parameters without manual sorts | Modern utilities handle sorting |
| `torch.no_grad()` context via decorator `@torch.no_grad()` (OK) but missing around inference | Ensure eval/inference paths wrap forward passes in `torch.no_grad()` | Prevents autograd tracking |
| `nn.Sequential(OrderedDict([...]))` with explicit `nn.LogSoftmax()` final layer | Often `nn.Sequential(..., nn.LogSoftmax(dim=1))` or use `nn.CrossEntropyLoss` which expects raw logits | CrossEntropy handles log-softmax internally |
| CPU/GPU branching via `if torch.cuda.is_available(): tensor = tensor.cuda()` | Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and `.to(device)` | Centralizes device handling |

## Review Flow

1. **Search for deprecated constructs** – look for `Variable`, `.data`, legacy
   tensor constructors, or functional calls missing keyword arguments such as
   `dim`.
2. **Check optimizers** – gradients should be zeroed via `optimizer.zero_grad()`
   and no manual `.data` mutations should remain.
3. **Verify tensor creation** – prefer `torch.tensor`/`torch.zeros` with explicit
   dtype/device rather than calling class constructors directly.
4. **Device handling** – confirm that tensors and models move via `.to(device)`
   instead of scattered `.cuda()` calls.
5. **Inference paths** – ensure evaluation code uses `model.eval()` and
   `with torch.no_grad()` where appropriate.

Keeping this sheet alongside the live PyTorch results makes it easy to confirm
that each upgraded block follows current PyTorch recommendations.
