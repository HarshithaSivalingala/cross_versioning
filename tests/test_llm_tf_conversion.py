import ast
import builtins
import json
import keyword
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

import src.agentic_upgrader as agentic_upgrader
import src.llm_interface as llm_interface
import src.utils as utils


def _normalize(code: str) -> str:
    return textwrap.dedent(code).strip()


_BUILTIN_NAMES = set(dir(builtins))


class _VariableNameNormalizer(ast.NodeTransformer):
    def __init__(self):
        self._name_map = {}
        self._counter = 0

    def _canonical_name(self, name):
        if name is None:
            return None
        if keyword.iskeyword(name) or name in _BUILTIN_NAMES:
            return name
        if name not in self._name_map:
            self._counter += 1
            self._name_map[name] = f"VAR{self._counter}"
        return self._name_map[name]

    def visit_Name(self, node):
        node.id = self._canonical_name(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._canonical_name(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        node.name = self._canonical_name(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.name = self._canonical_name(node.name)
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.arg = self._canonical_name(node.arg)
        return node

    def visit_alias(self, node):
        node.asname = self._canonical_name(node.asname)
        return node

    def visit_Global(self, node):
        node.names = [self._canonical_name(name) for name in node.names]
        return node

    def visit_Nonlocal(self, node):
        node.names = [self._canonical_name(name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node):
        node.name = self._canonical_name(node.name)
        return self.generic_visit(node)


def _canonicalize_for_comparison(code: str) -> str:
    normalized_code = _normalize(code)
    try:
        tree = ast.parse(normalized_code)
    except SyntaxError:
        return normalized_code
    canonical_tree = _VariableNameNormalizer().visit(tree)
    ast.fix_missing_locations(canonical_tree)
    return ast.dump(canonical_tree, include_attributes=False)


def _default_dataset_path(framework: str) -> Path:
    return Path(__file__).with_name(f"{framework}_conversion_dataset.json")


def _default_live_results_path(framework: str) -> Path:
    return Path(__file__).with_name(f"llm_live_results_{framework}.txt")


def _dataset_env_vars(framework: str, config) -> list:
    env_vars = [f"LLM_DATASET_PATH_{framework.upper()}"]
    env_vars.extend(config.get("dataset_env_vars", []))
    return env_vars


def _dataset_path_for_framework(framework: str, config) -> Path:
    for env_var in _dataset_env_vars(framework, config):
        override = os.getenv(env_var)
        if override:
            return Path(override)

    default_path = _default_dataset_path(framework)
    if default_path.exists():
        return default_path

    for legacy_name in config.get("legacy_dataset_filenames", []):
        legacy_path = Path(__file__).with_name(legacy_name)
        if legacy_path.exists():
            return legacy_path

    return default_path


def _live_results_path_for_framework(framework: str, config) -> Path:
    env_var = f"LLM_LIVE_RESULTS_PATH_{framework.upper()}"
    override = os.getenv(env_var)
    if override:
        return Path(override)

    default_path = _default_live_results_path(framework)
    if default_path.exists():
        return default_path

    for legacy_name in config.get("legacy_live_results_filenames", []):
        legacy_path = Path(__file__).with_name(legacy_name)
        if legacy_path.exists():
            return legacy_path

    return default_path


FRAMEWORK_DATASETS = {
    "tensorflow": {
        "source_key": "tf1",
        "target_key": "tf2",
        "input_label": "TF1",
        "output_label": "TF2",
        "dataset_env_vars": ["TF_CONVERSION_DATASET_PATH"],
        "legacy_dataset_filenames": ["tf_conversion_dataset.json"],
        "legacy_live_results_filenames": ["tf_llm_live_results.txt"],
    },
    "pytorch": {
        "source_key": "torch1",
        "target_key": "torch2",
        "input_label": "PyTorch legacy",
        "output_label": "PyTorch modern",
        "dataset_env_vars": ["TORCH_CONVERSION_DATASET_PATH"],
        "legacy_dataset_filenames": ["torch_conversion_dataset.json"],
    },
}


def _load_code_pairs(dataset_path: Path, config):
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_key = config["source_key"]
    target_key = config["target_key"]
    normalized = []
    for entry in raw_data:
        if source_key not in entry or target_key not in entry:
            raise KeyError(f"Dataset entry {entry.get('id')} missing required keys {source_key}/{target_key}")
        normalized.append({
            "id": entry["id"],
            "category": entry.get("category", ""),
            "input_code": _normalize(entry[source_key]),
            "expected_code": _normalize(entry[target_key]),
            "expected_changes": entry.get("expected_changes") or [],
        })
    return normalized


def _load_framework_samples():
    samples = []
    for framework, config in FRAMEWORK_DATASETS.items():
        dataset_path = _dataset_path_for_framework(framework, config)
        for sample in _load_code_pairs(dataset_path, config):
            samples.append((framework, sample))
    return samples


def _framework_sample_id(param):
    framework, sample = param
    return f"{framework}:{sample['id']}"


FRAMEWORK_SAMPLES = _load_framework_samples()


_ENFORCED_API_KEYWORDS = [
    "tf.session",
    "tf.placeholder",
    "np.asscalar",
    "torch.cuda.floattensor",
    "tf.get_variable",
    "tf.layers",
    "tf.contrib",
    "np.int",
    "np.float",
    "torch.autograd.variable",
]


def _change_recorded(expected_change: str, api_changes):
    lowered_expected = expected_change.lower()
    keywords = [keyword for keyword in _ENFORCED_API_KEYWORDS if keyword in lowered_expected]
    if not keywords:
        return True
    for change in api_changes:
        lowered_change = change.lower()
        if any(keyword in lowered_change for keyword in keywords):
            return True
    return False


@pytest.fixture(autouse=True)
def stub_tensorflow(monkeypatch):
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__path__ = []

    compat_stub = types.ModuleType("tensorflow.compat")
    compat_v1_stub = types.ModuleType("tensorflow.compat.v1")
    compat_stub.v1 = compat_v1_stub
    tf_stub.compat = compat_stub

    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.compat", compat_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.compat.v1", compat_v1_stub)


@pytest.fixture(autouse=True)
def stub_pytorch(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.__path__ = []

    nn_stub = types.ModuleType("torch.nn")
    nn_functional_stub = types.ModuleType("torch.nn.functional")
    nn_stub.functional = nn_functional_stub
    autograd_stub = types.ModuleType("torch.autograd")
    autograd_stub.Variable = type("Variable", (), {})
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.FloatTensor = type("FloatTensor", (), {})
    optim_stub = types.ModuleType("torch.optim")

    torch_stub.nn = nn_stub
    torch_stub.autograd = autograd_stub
    torch_stub.cuda = cuda_stub
    torch_stub.optim = optim_stub

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_stub)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", nn_functional_stub)
    monkeypatch.setitem(sys.modules, "torch.autograd", autograd_stub)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_stub)
    monkeypatch.setitem(sys.modules, "torch.optim", optim_stub)


@pytest.mark.parametrize("framework_sample", FRAMEWORK_SAMPLES, ids=_framework_sample_id)
def test_agentic_upgrader_accepts_known_pairs(tmp_path, monkeypatch, framework_sample):
    framework, sample = framework_sample
    input_path = tmp_path / f"{framework}_{sample['id']}.py"
    output_path = tmp_path / f"{framework}_{sample['id']}_converted.py"
    input_path.write_text(sample["input_code"] + "\n", encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        first_line = next((line for line in sample["input_code"].splitlines() if line.strip()), "")
        assert first_line in prompt
        return f"```python\n{sample['expected_code']}\n```"

    monkeypatch.setattr(agentic_upgrader.llm_interface, "call_llm", fake_llm)

    def fake_validate_code(path, *, run_runtime=True, preloaded_code=None):
        if preloaded_code is None:
            with open(path, "r", encoding="utf-8") as handle:
                code_to_check = handle.read()
        else:
            code_to_check = preloaded_code
        compile(code_to_check, path, "exec")
        return True, None

    monkeypatch.setattr(agentic_upgrader.validator, "validate_code", fake_validate_code)

    result = agentic_upgrader.upgrade_file(str(input_path), str(output_path))
    assert result.success, f"{framework}:{sample['id']} failed with: {result.error}"

    upgraded_code = output_path.read_text(encoding="utf-8").strip()
    assert upgraded_code == sample["expected_code"]

    expected_changes = sample.get("expected_changes") or []
    for change in expected_changes:
        assert _change_recorded(change, result.api_changes), f"Missing expected change '{change}' for {framework}:{sample['id']}"

    assert isinstance(result.api_changes, list)


def test_upgrade_file_retries_when_tf1_patterns_detected(tmp_path, monkeypatch):
    input_path = tmp_path / "example.py"
    output_path = tmp_path / "example_out.py"
    input_path.write_text("import tensorflow as tf\nprint('legacy')\n", encoding="utf-8")

    responses = [
        """```python
import tensorflow as tf
with tf.compat.v1.Session() as sess:
    print(sess.run(tf.constant(1)))
```""",
        """```python
import tensorflow as tf
print(tf.constant(1).numpy())
```""",
    ]

    def fake_llm(prompt: str) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agentic_upgrader.llm_interface, "call_llm", fake_llm)

    def fake_validate_code(path, *, run_runtime=True, preloaded_code=None):
        code_to_check = preloaded_code or Path(path).read_text(encoding="utf-8")
        compile(code_to_check, path, "exec")
        return True, None

    monkeypatch.setattr(agentic_upgrader.validator, "validate_code", fake_validate_code)

    result = agentic_upgrader.upgrade_file(str(input_path), str(output_path))

    assert result.success is True
    assert result.attempts == 2
    upgraded_code = output_path.read_text(encoding="utf-8")
    assert "compat.v1" not in upgraded_code
    assert "Session" not in upgraded_code


RUN_LLM_EVAL = os.getenv("RUN_LLM_EVAL") == "1" or os.getenv("RUN_TF_LLM_EVAL") == "1"


@pytest.fixture(scope="session")
def live_llm_results_paths():
    if not RUN_LLM_EVAL:
        return {}
    paths = {}
    for framework, config in FRAMEWORK_DATASETS.items():
        path = _live_results_path_for_framework(framework, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        paths[framework] = path
    return paths


@pytest.mark.skipif(not RUN_LLM_EVAL, reason="Set RUN_LLM_EVAL=1 to enable live LLM verification")
@pytest.mark.parametrize("framework_sample", FRAMEWORK_SAMPLES, ids=_framework_sample_id)
def test_llm_live_conversion_matches_expected(framework_sample, live_llm_results_paths):
    framework, sample = framework_sample
    prompt = utils.build_prompt(sample["input_code"])
    response = llm_interface.call_llm(prompt)
    upgraded = _normalize(agentic_upgrader.clean_llm_response(response))
    results_path = live_llm_results_paths.get(framework)
    if results_path:
        labels = FRAMEWORK_DATASETS[framework]
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write("=" * 100 + "\n")
            handle.write(f"ID       : {sample['id']}\n")
            handle.write(f"Framework: {framework}\n")
            handle.write(f"Category : {sample['category']}\n")
            handle.write("=" * 100 + "\n\n")

            handle.write(f"INPUT ({labels['input_label']})\n-----------\n```python\n")
            handle.write(sample["input_code"])
            handle.write("\n```\n\n")

            handle.write(f"EXPECTED ({labels['output_label']})\n--------------\n```python\n")
            handle.write(sample["expected_code"])
            handle.write("\n```\n\n")

            handle.write("LLM OUTPUT\n----------\n```python\n")
            handle.write(upgraded)
            handle.write("\n```\n\n")

            handle.write("RAW RESPONSE\n------------\n")
            handle.write(response.strip())
            handle.write("\n\n")
    assert _canonicalize_for_comparison(upgraded) == _canonicalize_for_comparison(sample["expected_code"]), sample["id"]
