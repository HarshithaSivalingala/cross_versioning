import os
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class FileUpgradeResult:
    """Result of upgrading a single file"""
    file_path: str
    success: bool
    attempts: int
    api_changes: List[str]
    error: str = None
    diff: str = None

class UpgradeReportGenerator:
    """Generate detailed upgrade reports"""
    
    def __init__(self):
        self.results: List[FileUpgradeResult] = []
        self.dependency_changes: List[str] = []
        self.start_time = datetime.now()
    
    def add_file_result(self, result: FileUpgradeResult):
        """Add a file upgrade result"""
        self.results.append(result)
    
    def add_dependency_changes(self, changes: List[str]):
        """Add dependency update changes"""
        self.dependency_changes.extend(changes)
    
    def generate_report(self, output_path: str) -> None:
        """Generate markdown upgrade report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        report = f"""# ML Repository Upgrade Report

**Generated:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Duration:** {duration.total_seconds():.1f} seconds  
**Total Files:** {len(self.results)}  
**Successful:** {len(successful)}  
**Failed:** {len(failed)}  

## Summary

{len(successful)}/{len(self.results)} files upgraded successfully ({len(successful)/len(self.results)*100:.1f}%).

"""

        # Dependency updates
        if self.dependency_changes:
            report += "## Dependency Updates\n\n"
            for change in self.dependency_changes:
                report += f"- {change}\n"
            report += "\n"

        # Successful upgrades
        if successful:
            report += "## ✅ Successfully Upgraded Files\n\n"
            for result in successful:
                report += f"### `{result.file_path}`\n\n"
                report += f"- **Attempts:** {result.attempts}\n"
                
                if result.api_changes:
                    report += "- **API Changes:**\n"
                    for change in result.api_changes:
                        report += f"  - {change}\n"
                
                if result.diff:
                    report += "\n**Changes:**\n```diff\n"
                    # Limit diff to first 20 lines to keep report readable
                    diff_lines = result.diff.split('\n')[:20]
                    report += '\n'.join(diff_lines)
                    if len(result.diff.split('\n')) > 20:
                        diff_lines_count = len(result.diff.split('\n')) - 20
                        report += f"\n... ({diff_lines_count} more lines)"
                    report += "\n```\n\n"
                else:
                    report += "\n"

        # Failed upgrades
        if failed:
            report += "## ❌ Failed Upgrades\n\n"
            for result in failed:
                report += f"### `{result.file_path}`\n\n"
                report += f"- **Attempts:** {result.attempts}\n"
                report += f"- **Error:** {result.error}\n\n"

        # Statistics
        report += "## 📊 Statistics\n\n"
        total_attempts = sum(r.attempts for r in self.results)
        avg_attempts = total_attempts / len(self.results) if self.results else 0
        
        report += f"- **Average attempts per file:** {avg_attempts:.1f}\n"
        report += f"- **Total LLM calls:** {total_attempts}\n"
        
        # API change frequency
        all_changes = []
        for result in successful:
            all_changes.extend(result.api_changes)
        
        if all_changes:
            change_counts = {}
            for change in all_changes:
                change_counts[change] = change_counts.get(change, 0) + 1
            
            report += "\n**Most common API changes:**\n"
            sorted_changes = sorted(change_counts.items(), key=lambda x: x[1], reverse=True)
            for change, count in sorted_changes[:5]:
                report += f"- {change} ({count} files)\n"

        # Manual review section
        if failed:
            report += "\n## Manual Review Needed\n\n"
            report += "The following files failed automatic upgrade and require manual attention:\n\n"
            for i, result in enumerate(failed, 1):
                short_path = os.path.basename(result.file_path)
                report += f"{i}. **`{short_path}`** - {result.error}\n"

        # Recommendations
        report += "\n## 💡 Recommendations\n\n"
        if successful:
            report += f"1. **Immediate Use**: The {len(successful)} successfully upgraded files are ready to use with modern ML libraries\n"
        report += "2. **Testing**: Run your existing test suite to verify functionality\n"
        if failed:
            report += "3. **Manual Migration**: Review failed files for manual upgrade opportunities\n"
        report += "4. **Incremental Adoption**: Consider upgrading successfully migrated modules first\n"

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
