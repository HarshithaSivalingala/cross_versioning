import argparse
import os
import sys
from dotenv import load_dotenv
import repo_upgrader 

load_dotenv()

def main():
    """Command line interface for ML Repository Upgrader"""
    parser = argparse.ArgumentParser(
        description="Upgrade ML repositories to use latest APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ml-upgrader old_repo/ new_repo/
  ml-upgrader --input old_repo.zip --output upgraded_repo.zip
  ml-upgrader tensorflow_project/ modern_tf_project/ --model openai/gpt-4
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input repository or .zip file"
    )
    
    parser.add_argument(
        "output_path", 
        help="Path for upgraded repository"
    )
    
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        choices=["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4"],
        help="LLM model to use (default: openai/gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per file (default: 5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path '{args.input_path}' does not exist")
        sys.exit(1)
    
    # Handle .zip files
    if args.input_path.endswith('.zip'):
        print("📦 Extracting .zip file...")
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        extract_path = os.path.join(temp_dir, "extracted")
        
        with zipfile.ZipFile(args.input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        args.input_path = extract_path
    
    print(f"🚀 Starting upgrade: {args.input_path} → {args.output_path}")
    
    try:
        # Set global variables for configuration
        os.environ["ML_UPGRADER_MODEL"] = args.model
        os.environ["ML_UPGRADER_MAX_RETRIES"] = str(args.max_retries)
        
        report_path = repo_upgrader.upgrade_repo(args.input_path, args.output_path)
        
        print(f"✅ Upgrade completed successfully!")
        print(f"📄 Report: {report_path}")
        
    except Exception as e:
        print(f"❌ Upgrade failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
