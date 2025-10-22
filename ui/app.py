import streamlit as st
import zipfile
import os
import shutil
import tempfile
import sys
import json
from dotenv import load_dotenv

# Simple path fix
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Direct imports
import repo_upgrader

load_dotenv()


def _parse_runtime_command(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(parsed, str):
        parsed = parsed.strip()
        if not parsed:
            raise ValueError("Runtime command string cannot be empty.")
        return parsed

    if isinstance(parsed, list):
        command_parts = []
        for item in parsed:
            if isinstance(item, (str, int, float)):
                command_parts.append(str(item))
            else:
                raise ValueError("Runtime command list items must be strings or numbers.")
        if not command_parts:
            raise ValueError("Runtime command list cannot be empty.")
        return command_parts

    raise ValueError("Runtime command must be a JSON string or list of strings.")


def _parse_runtime_env(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Runtime environment must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError('Runtime environment must be a JSON object, e.g. {"PYTHONPATH": "src"}.')

    normalized = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError("Runtime environment keys must be strings.")
        normalized[key] = "" if value is None else str(value)
    return normalized


def _parse_setup_commands(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Interpret as newline-separated shell commands
        return [line.strip() for line in raw.splitlines() if line.strip()]

    if not isinstance(parsed, list):
        raise ValueError("Setup commands must be a JSON array.")

    normalized = []
    for idx, entry in enumerate(parsed):
        label = f"Setup command #{idx + 1}"
        if isinstance(entry, str):
            stripped = entry.strip()
            if not stripped:
                raise ValueError(f"{label} cannot be empty.")
            normalized.append(stripped)
        elif isinstance(entry, (list, tuple)):
            command_parts = []
            for part in entry:
                if isinstance(part, (str, int, float)):
                    command_parts.append(str(part))
                else:
                    raise ValueError(f"{label} list items must be strings or numbers.")
            if not command_parts:
                raise ValueError(f"{label} list cannot be empty.")
            normalized.append(command_parts)
        elif isinstance(entry, (int, float)):
            normalized.append(str(entry))
        else:
            raise ValueError(f"{label} must be a string or list of strings.")

    return normalized

def main():
    st.set_page_config(
        page_title="ML Repo Upgrader", 
        page_icon="🔄", 
        layout="wide"
    )
    
    st.title("🔄 ML Repository Upgrader")
    st.markdown("""
    **Automatically upgrade repositories to use the latest APIs for:**
    - TensorFlow (1.x → 2.x)
    - PyTorch (legacy → modern)
    - NumPy (deprecated functions)
    - JAX (API updates)
    """)
    
    # Sidebar for settings
    runtime_ui_state = {
        "enabled": False,
        "command": '["python", "main.py"]',
        "timeout": 180,
        "skip_install": False,
        "force_reinstall": False,
        "shell": False,
        "cwd": "",
        "env": '{"PYTHONPATH": "src"}',
        "max_log_chars": 6000,
        "setup_commands": "",
    }

    with st.sidebar:
        st.header("⚙️ Settings")
        
        existing_api_key = os.getenv("OPENROUTER_API_KEY", "")
        api_key_input = st.text_input(
            "OpenRouter API Key (optional)",
            type="password",
            help="Leave blank to use the key from your .env file.",
        )

        if api_key_input:
            os.environ["OPENROUTER_API_KEY"] = api_key_input
            st.success("✅ API key set for this session")
        elif existing_api_key:
            st.info("Using OPENROUTER_API_KEY from environment.")
        else:
            st.warning("⚠️ Provide an OpenRouter key via .env or enter it here before running an upgrade.")

        model_options = ["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4"]
        model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help="Choose the OpenRouter model; defaults to openai/gpt-4o-mini.",
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_retries = st.slider("Max retries per file", 1, 10, 5)
            os.environ["ML_UPGRADER_MAX_RETRIES"] = str(max_retries)
            
            show_progress = st.checkbox("Show detailed progress", True)

            runtime_ui_state["enabled"] = st.checkbox(
                "Enable runtime validation",
                value=runtime_ui_state["enabled"],
                help="Run a quick command (e.g. smoke tests) after each upgrade attempt."
            )

            if runtime_ui_state["enabled"]:
                runtime_ui_state["command"] = st.text_area(
                    "Runtime command (string or JSON list)",
                    value=runtime_ui_state["command"],
                    help='Examples: "python main.py" or ["python", "main.py"]'
                )
                runtime_ui_state["timeout"] = st.number_input(
                    "Runtime timeout (seconds)",
                    min_value=1,
                    max_value=3600,
                    value=int(runtime_ui_state["timeout"]),
                    step=10
                )
                runtime_ui_state["skip_install"] = st.checkbox(
                    "Skip dependency installation",
                    value=runtime_ui_state["skip_install"]
                )
                runtime_ui_state["force_reinstall"] = st.checkbox(
                    "Force reinstall dependencies",
                    value=runtime_ui_state["force_reinstall"]
                )
                runtime_ui_state["shell"] = st.checkbox(
                    "Use shell execution",
                    value=runtime_ui_state["shell"],
                    help="Enable when the command must run through the shell."
                )
                runtime_ui_state["cwd"] = st.text_input(
                    "Working directory (optional)",
                    value=runtime_ui_state["cwd"],
                    help="Relative path from the repository root."
                )
                runtime_ui_state["env"] = st.text_area(
                    "Environment variables (JSON)",
                    value=runtime_ui_state["env"],
                    help='Example: {"PYTHONPATH": "src"}'
                )
                runtime_ui_state["setup_commands"] = st.text_area(
                    "Setup commands (optional)",
                    value=runtime_ui_state["setup_commands"],
                    help="Commands to run before the runtime command. Use newline-separated shell commands or a JSON array (e.g. [\"wget ...\", [\"python\", \"scripts/setup.py\"]])."
                )
                runtime_ui_state["max_log_chars"] = st.number_input(
                    "Max runtime log characters",
                    min_value=0,
                    max_value=20000,
                    value=int(runtime_ui_state["max_log_chars"]),
                    step=500
                )
            else:
                st.caption("Runtime validation disabled. Enable above to mirror ml_upgrader_runtime.json settings.")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Repository")
        
        uploaded_file = st.file_uploader(
            "Upload repository (.zip)", 
            type=["zip"],
            help="Upload a .zip file containing your ML repository"
        )
        
        if uploaded_file and not os.getenv("OPENROUTER_API_KEY"):
            st.error("❌ Please set an OpenRouter API key in the sidebar or .env before running an upgrade.")
            return

        if uploaded_file and os.getenv("OPENROUTER_API_KEY"):
            # Create temp directories
            temp_dir = tempfile.mkdtemp()
            old_repo_path = os.path.join(temp_dir, "old_repo")
            new_repo_path = os.path.join(temp_dir, "new_repo")
            
            try:
                # Extract uploaded zip
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(old_repo_path)
                
                st.success("✅ Repository uploaded and extracted")
                
                # Show repository structure
                st.subheader("📂 Repository Structure")
                python_files = []
                for root, dirs, files in os.walk(old_repo_path):
                    for file in files:
                        if file.endswith('.py'):
                            rel_path = os.path.relpath(os.path.join(root, file), old_repo_path)
                            python_files.append(rel_path)
                
                st.write(f"Found **{len(python_files)}** Python files:")
                with st.expander("View files"):
                    for file in python_files[:10]:  # Show first 10
                        st.text(f"📄 {file}")
                    if len(python_files) > 10:
                        st.text(f"... and {len(python_files) - 10} more files")
                
                # Upgrade button
                if st.button("🚀 Start Upgrade", type="primary", use_container_width=True):

                    runtime_config_payload = None
                    if runtime_ui_state["enabled"]:
                        try:
                            runtime_command = _parse_runtime_command(runtime_ui_state["command"])
                        except ValueError as exc:
                            st.error(f"Runtime command error: {exc}")
                            return

                        if runtime_command is None:
                            st.error("Runtime command is required when runtime validation is enabled.")
                            return

                        try:
                            runtime_env = _parse_runtime_env(runtime_ui_state["env"])
                        except ValueError as exc:
                            st.error(f"Runtime environment error: {exc}")
                            return

                        try:
                            setup_commands = _parse_setup_commands(runtime_ui_state["setup_commands"])
                        except ValueError as exc:
                            st.error(f"Setup commands error: {exc}")
                            return

                        runtime_config_payload = {
                            "command": runtime_command,
                            "timeout": int(runtime_ui_state["timeout"]),
                            "skip_install": bool(runtime_ui_state["skip_install"]),
                            "force_reinstall": bool(runtime_ui_state["force_reinstall"]),
                            "shell": bool(runtime_ui_state["shell"]),
                            "max_log_chars": int(runtime_ui_state["max_log_chars"]),
                            "env": runtime_env,
                        }

                        runtime_cwd = (runtime_ui_state["cwd"] or "").strip()
                        if runtime_cwd:
                            runtime_config_payload["cwd"] = runtime_cwd
                        if setup_commands:
                            runtime_config_payload["setup_commands"] = setup_commands
                    
                    with st.spinner("🔄 Upgrading repository... This may take a few minutes."):
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Start upgrade
                        status_text.text("📦 Updating dependencies...")
                        progress_bar.progress(10)
                        
                        try:
                            # Set model
                            os.environ["ML_UPGRADER_MODEL"] = model

                            runtime_config_path = None
                            previous_runtime_config_env = os.getenv("ML_UPGRADER_RUNTIME_CONFIG")
                            try:
                                if runtime_config_payload is not None:
                                    runtime_temp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
                                    json.dump(runtime_config_payload, runtime_temp, indent=2)
                                    runtime_temp.flush()
                                    runtime_temp.close()
                                    runtime_config_path = runtime_temp.name
                                    os.environ["ML_UPGRADER_RUNTIME_CONFIG"] = runtime_config_path

                                report_path = repo_upgrader.upgrade_repo(old_repo_path, new_repo_path)
                            finally:
                                if runtime_config_path and os.path.exists(runtime_config_path):
                                    os.unlink(runtime_config_path)
                                if previous_runtime_config_env is not None:
                                    os.environ["ML_UPGRADER_RUNTIME_CONFIG"] = previous_runtime_config_env
                                elif runtime_config_payload is not None:
                                    os.environ.pop("ML_UPGRADER_RUNTIME_CONFIG", None)

                            progress_bar.progress(90)
                            
                            status_text.text("📄 Generating report...")
                            
                            # Create downloadable zip
                            output_zip = os.path.join(temp_dir, "upgraded_repo.zip")
                            shutil.make_archive(output_zip[:-4], 'zip', new_repo_path)
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Upgrade complete!")
                            
                            st.success("🎉 Repository upgraded successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Upgrade failed: {str(e)}")
                            return
                    
                    # Results section
                    with col2:
                        st.subheader("📊 Results")
                        
                        # Show upgrade report
                        if os.path.exists(report_path):
                            with open(report_path, 'r') as f:
                                report_content = f.read()
                            
                            # Extract summary stats
                            if "**Successful:**" in report_content:
                                lines = report_content.split('\n')
                                success_count = "0"
                                failed_count = "0"
                                for line in lines:
                                    if "**Successful:**" in line:
                                        success_count = line.split('**Successful:** ')[1].strip()
                                    if "**Failed:**" in line:
                                        failed_count = line.split('**Failed:** ')[1].strip()
                                
                                col_s, col_f = st.columns(2)
                                with col_s:
                                    st.metric("✅ Successfully Upgraded", success_count)
                                with col_f:
                                    st.metric("❌ Failed", failed_count)
                            
                            # Show report preview
                            st.subheader("📄 Upgrade Report Preview")
                            with st.expander("View Full Report"):
                                st.markdown(report_content)
                        
                        # Download buttons
                        st.subheader("📥 Downloads")
                        
                        # Download upgraded repository
                        if os.path.exists(output_zip):
                            with open(output_zip, "rb") as f:
                                st.download_button(
                                    "📦 Download Upgraded Repository",
                                    f.read(),
                                    file_name="upgraded_repo.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                        
                        # Download report only
                        if os.path.exists(report_path):
                            with open(report_path, "r") as f:
                                st.download_button(
                                    "📄 Download Upgrade Report",
                                    f.read(),
                                    file_name="UPGRADE_REPORT.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
            
            except Exception as e:
                st.error(f"Error processing upload: {str(e)}")
            
            finally:
                # Cleanup
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    Upload your legacy ML repository as a .zip file
    """)

if __name__ == "__main__":
    main()
