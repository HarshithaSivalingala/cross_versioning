
import llm_interface
import validator
import utils
import report_generator
import os

def upgrade_file(input_path: str, output_path: str) -> report_generator.FileUpgradeResult:
    """Upgrade a single file with detailed tracking"""
    
    MAX_RETRIES = int(os.getenv("ML_UPGRADER_MAX_RETRIES", "5"))
    
    if not os.path.exists(input_path):
        return report_generator.FileUpgradeResult(
            file_path=input_path,
            success=False,
            attempts=0,
            api_changes=[],
            error="Input file not found"
        )

    skip_reason = utils.should_skip_for_upgrade(input_path)
    if skip_reason:
        print(f"ℹ️ Skipping {input_path}: {skip_reason}")
        return report_generator.FileUpgradeResult(
            file_path=input_path,
            success=False,
            attempts=0,
            api_changes=[],
            error=skip_reason
        )
    
    old_code = utils.read_file(input_path)
    error = None
    current_code = old_code

    try:
        precheck_valid, precheck_error = validator.validate_code(input_path)
        if not precheck_valid:
            error = precheck_error
    except Exception as exc:  # pragma: no cover - defensive, mirrors runtime loop handling
        error = str(exc)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = utils.build_prompt(current_code, error)
            response = llm_interface.call_llm(prompt)
            new_code = clean_llm_response(response)

            stripped_code = new_code.strip()
            if not stripped_code:
                error = "LLM returned empty response"
                print(f"⚠️ {input_path} attempt {attempt} error: {error}")
                continue

            apology_prefixes = ("i'm sorry", "im sorry", "sorry", "i cannot", "i can’t")
            if stripped_code.lower().startswith(apology_prefixes) or stripped_code.startswith("# upgraded code here"):
                error = "LLM returned placeholder text instead of upgraded code"
                print(f"⚠️ {input_path} attempt {attempt} error: {error}")
                continue

            utils.write_file(output_path, new_code)
            
            # Validate the new code
            is_valid, error = validator.validate_code(output_path)
            
            if is_valid:
                # Success! Generate final result
                api_changes = utils.extract_api_changes(old_code, new_code)
                diff = utils.generate_diff(old_code, new_code, os.path.basename(input_path))
                
                print(f"✅ {input_path} upgraded successfully in {attempt} attempts")
                
                return report_generator.FileUpgradeResult(
                    file_path=input_path,
                    success=True,
                    attempts=attempt,
                    api_changes=api_changes,
                    diff=diff
                )
            
            # If validation failed, use the new code for next iteration
            current_code = new_code
            print(f"⚠️ {input_path} attempt {attempt} failed: {error}")
            
        except Exception as e:
            error = str(e)
            print(f"⚠️ {input_path} attempt {attempt} error: {error}")

    # All attempts failed
    print(f"❌ Failed to upgrade {input_path} after {MAX_RETRIES} attempts")
    
    return report_generator.FileUpgradeResult(
        file_path=input_path,
        success=False,
        attempts=MAX_RETRIES,
        api_changes=[],
        error=error or "Maximum retries exceeded"
    )


import re

def clean_llm_response(response: str) -> str:
    """Extract the upgraded Python code from LLM response (strip markdown, explanations)"""
    # Extract text inside the first ```python ... ```
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
