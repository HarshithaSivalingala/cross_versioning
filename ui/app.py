import streamlit as st
import zipfile
import os
import shutil
import tempfile
from pathlib import Path
import sys

# Simple path fix
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Direct imports
import repo_upgrader

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
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Required for LLM-powered code upgrades"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API key set")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
        
        # Model selection
        model = st.selectbox(
    "Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
    help="Choose Gemini model (flash is faster/cheaper)"
)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_retries = st.slider("Max retries per file", 1, 10, 5)
            os.environ["ML_UPGRADER_MAX_RETRIES"] = str(max_retries)
            
            show_progress = st.checkbox("Show detailed progress", True)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Repository")
        
        uploaded_file = st.file_uploader(
            "Upload repository (.zip)", 
            type=["zip"],
            help="Upload a .zip file containing your ML repository"
        )
        
        if uploaded_file and not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar first!")
            return
        
        if uploaded_file and api_key:
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
                            
                            report_path = repo_upgrader.upgrade_repo(old_repo_path, new_repo_path)
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