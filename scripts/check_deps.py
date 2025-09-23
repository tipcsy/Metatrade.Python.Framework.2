#!/usr/bin/env python3
"""
Dependency checker script for MetaTrader Python Framework.

This script checks for dependency conflicts, security vulnerabilities,
outdated packages, and compatibility issues.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pkg_resources


class DependencyChecker:
    """Dependency checker and analyzer."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize dependency checker.

        Args:
            project_root: Project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = project_root

        self.colors = {
            "RED": "\033[0;31m",
            "GREEN": "\033[0;32m",
            "YELLOW": "\033[0;33m",
            "BLUE": "\033[0;34m",
            "PURPLE": "\033[0;35m",
            "CYAN": "\033[0;36m",
            "WHITE": "\033[0;37m",
            "RESET": "\033[0m",
        }

    def print_colored(self, message: str, color: str = "WHITE") -> None:
        """Print colored message."""
        color_code = self.colors.get(color.upper(), self.colors["WHITE"])
        print(f"{color_code}{message}{self.colors['RESET']}")

    def print_header(self, title: str) -> None:
        """Print section header."""
        self.print_colored("\n" + "=" * 60, "CYAN")
        self.print_colored(f" {title}", "CYAN")
        self.print_colored("=" * 60, "CYAN")

    def get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages with versions."""
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name.lower()] = dist.version
        return packages

    def parse_requirements_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Parse requirements file and extract package names and versions.

        Args:
            file_path: Path to requirements file

        Returns:
            List of (package_name, version_spec) tuples
        """
        requirements = []

        if not file_path.exists():
            return requirements

        try:
            with file_path.open('r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Skip -r includes for now
                if line.startswith('-r'):
                    continue

                # Parse package specification
                if '>=' in line:
                    package, version = line.split('>=', 1)
                    requirements.append((package.strip(), f">={version.strip()}"))
                elif '==' in line:
                    package, version = line.split('==', 1)
                    requirements.append((package.strip(), f"=={version.strip()}"))
                elif '>' in line:
                    package, version = line.split('>', 1)
                    requirements.append((package.strip(), f">{version.strip()}"))
                else:
                    requirements.append((line.strip(), ""))

        except Exception as e:
            self.print_colored(f"Error parsing {file_path}: {e}", "RED")

        return requirements

    def check_requirements_consistency(self) -> Dict[str, List[str]]:
        """Check consistency across different requirements files."""
        self.print_header("Checking Requirements Consistency")

        requirements_dir = self.project_root / "requirements"
        files = ["base.txt", "development.txt", "testing.txt", "production.txt"]

        all_requirements = {}
        conflicts = {}

        for file_name in files:
            file_path = requirements_dir / file_name
            if file_path.exists():
                requirements = self.parse_requirements_file(file_path)
                all_requirements[file_name] = requirements

                for package, version in requirements:
                    package_lower = package.lower()

                    if package_lower not in conflicts:
                        conflicts[package_lower] = []

                    conflicts[package_lower].append((file_name, version))

        # Find actual conflicts
        real_conflicts = {}
        for package, versions in conflicts.items():
            if len(set(v[1] for v in versions if v[1])) > 1:
                real_conflicts[package] = versions

        if real_conflicts:
            self.print_colored("❌ Version conflicts found:", "RED")
            for package, versions in real_conflicts.items():
                self.print_colored(f"  {package}:", "YELLOW")
                for file_name, version in versions:
                    self.print_colored(f"    {file_name}: {version or 'any'}", "WHITE")
        else:
            self.print_colored("✅ No version conflicts found", "GREEN")

        return real_conflicts

    def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages."""
        self.print_header("Checking for Outdated Packages")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )

            outdated = json.loads(result.stdout)

            if outdated:
                self.print_colored(f"❌ Found {len(outdated)} outdated packages:", "YELLOW")
                for package in outdated:
                    self.print_colored(
                        f"  {package['name']}: {package['version']} -> {package['latest_version']}",
                        "WHITE"
                    )
            else:
                self.print_colored("✅ All packages are up to date", "GREEN")

            return outdated

        except subprocess.CalledProcessError as e:
            self.print_colored(f"❌ Error checking outdated packages: {e}", "RED")
            return []
        except json.JSONDecodeError as e:
            self.print_colored(f"❌ Error parsing pip output: {e}", "RED")
            return []

    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety."""
        self.print_header("Checking for Security Vulnerabilities")

        try:
            # Try to run safety check
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit
            )

            if result.returncode == 0:
                self.print_colored("✅ No known security vulnerabilities found", "GREEN")
                return []

            # Parse vulnerabilities
            try:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    self.print_colored(f"❌ Found {len(vulnerabilities)} security vulnerabilities:", "RED")
                    for vuln in vulnerabilities:
                        self.print_colored(f"  {vuln.get('package_name', 'Unknown')}: {vuln.get('vulnerability_id', 'Unknown')}", "RED")
                        self.print_colored(f"    {vuln.get('advisory', 'No description')}", "WHITE")
                return vulnerabilities
            except json.JSONDecodeError:
                self.print_colored("❌ Error parsing safety output", "RED")
                return []

        except FileNotFoundError:
            self.print_colored("❌ Safety not installed. Install with: pip install safety", "YELLOW")
            return []
        except Exception as e:
            self.print_colored(f"❌ Error running safety check: {e}", "RED")
            return []

    def check_pip_audit(self) -> List[Dict]:
        """Check for vulnerabilities using pip-audit."""
        self.print_header("Checking Vulnerabilities with pip-audit")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "--format=json"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                self.print_colored("✅ No vulnerabilities found by pip-audit", "GREEN")
                return []

            try:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    self.print_colored(f"❌ Found vulnerabilities:", "RED")
                    for vuln in vulnerabilities:
                        package = vuln.get('package', {})
                        self.print_colored(f"  {package.get('name', 'Unknown')} {package.get('version', '')}", "RED")
                return vulnerabilities
            except json.JSONDecodeError:
                self.print_colored("❌ Error parsing pip-audit output", "RED")
                return []

        except FileNotFoundError:
            self.print_colored("❌ pip-audit not installed. Install with: pip install pip-audit", "YELLOW")
            return []
        except Exception as e:
            self.print_colored(f"❌ Error running pip-audit: {e}", "RED")
            return []

    def check_dependency_conflicts(self) -> List[str]:
        """Check for dependency conflicts."""
        self.print_header("Checking Dependency Conflicts")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                self.print_colored("✅ No dependency conflicts found", "GREEN")
                return []
            else:
                conflicts = result.stdout.strip().split('\n')
                self.print_colored(f"❌ Found dependency conflicts:", "RED")
                for conflict in conflicts:
                    if conflict.strip():
                        self.print_colored(f"  {conflict}", "WHITE")
                return conflicts

        except Exception as e:
            self.print_colored(f"❌ Error checking dependencies: {e}", "RED")
            return []

    def check_package_licenses(self) -> Dict[str, str]:
        """Check package licenses."""
        self.print_header("Checking Package Licenses")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )

            licenses = json.loads(result.stdout)
            license_counts = {}
            unknown_licenses = []

            for package in licenses:
                license_name = package.get('License', 'Unknown')
                if license_name in ('Unknown', 'UNKNOWN', ''):
                    unknown_licenses.append(package.get('Name', 'Unknown'))
                else:
                    license_counts[license_name] = license_counts.get(license_name, 0) + 1

            self.print_colored("License distribution:", "BLUE")
            for license_name, count in sorted(license_counts.items()):
                self.print_colored(f"  {license_name}: {count} packages", "WHITE")

            if unknown_licenses:
                self.print_colored(f"❌ Packages with unknown licenses: {', '.join(unknown_licenses)}", "YELLOW")

            return license_counts

        except FileNotFoundError:
            self.print_colored("❌ pip-licenses not installed. Install with: pip install pip-licenses", "YELLOW")
            return {}
        except Exception as e:
            self.print_colored(f"❌ Error checking licenses: {e}", "RED")
            return {}

    def analyze_package_sizes(self) -> List[Tuple[str, int]]:
        """Analyze package sizes."""
        self.print_header("Analyzing Package Sizes")

        try:
            # Get package information
            installed_packages = self.get_installed_packages()
            package_sizes = []

            for package_name in installed_packages:
                try:
                    dist = pkg_resources.get_distribution(package_name)
                    if dist.location:
                        location_path = Path(dist.location)
                        if location_path.exists():
                            # Calculate directory size
                            total_size = sum(
                                f.stat().st_size
                                for f in location_path.rglob('*')
                                if f.is_file()
                            )
                            package_sizes.append((package_name, total_size))
                except Exception:
                    continue

            # Sort by size (largest first)
            package_sizes.sort(key=lambda x: x[1], reverse=True)

            self.print_colored("Largest packages:", "BLUE")
            for package, size in package_sizes[:10]:
                size_mb = size / (1024 * 1024)
                self.print_colored(f"  {package}: {size_mb:.1f} MB", "WHITE")

            return package_sizes

        except Exception as e:
            self.print_colored(f"❌ Error analyzing package sizes: {e}", "RED")
            return []

    def generate_dependency_report(self) -> Dict:
        """Generate comprehensive dependency report."""
        self.print_header("Generating Dependency Report")

        report = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "installed_packages": self.get_installed_packages(),
            "requirements_conflicts": self.check_requirements_consistency(),
            "outdated_packages": self.check_outdated_packages(),
            "security_vulnerabilities": self.check_security_vulnerabilities(),
            "dependency_conflicts": self.check_dependency_conflicts(),
            "package_licenses": self.check_package_licenses(),
            "package_sizes": self.analyze_package_sizes(),
        }

        # Save report
        report_file = self.project_root / "dependency-report.json"
        try:
            with report_file.open('w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            self.print_colored(f"✅ Report saved to {report_file}", "GREEN")
        except Exception as e:
            self.print_colored(f"❌ Error saving report: {e}", "RED")

        return report

    def run_all_checks(self) -> bool:
        """Run all dependency checks."""
        self.print_colored("\n🔍 MetaTrader Python Framework - Dependency Checker", "CYAN")

        # Run all checks
        conflicts = self.check_requirements_consistency()
        outdated = self.check_outdated_packages()
        security_vulns = self.check_security_vulnerabilities()
        pip_audit_vulns = self.check_pip_audit()
        dep_conflicts = self.check_dependency_conflicts()
        licenses = self.check_package_licenses()
        sizes = self.analyze_package_sizes()

        # Summary
        self.print_header("Summary")

        issues_found = False

        if conflicts:
            self.print_colored(f"❌ Requirements conflicts: {len(conflicts)}", "RED")
            issues_found = True
        else:
            self.print_colored("✅ No requirements conflicts", "GREEN")

        if outdated:
            self.print_colored(f"⚠️  Outdated packages: {len(outdated)}", "YELLOW")
        else:
            self.print_colored("✅ All packages up to date", "GREEN")

        if security_vulns:
            self.print_colored(f"❌ Security vulnerabilities: {len(security_vulns)}", "RED")
            issues_found = True
        else:
            self.print_colored("✅ No known security vulnerabilities", "GREEN")

        if dep_conflicts:
            self.print_colored(f"❌ Dependency conflicts: {len(dep_conflicts)}", "RED")
            issues_found = True
        else:
            self.print_colored("✅ No dependency conflicts", "GREEN")

        # Generate comprehensive report
        self.generate_dependency_report()

        if issues_found:
            self.print_colored("\n❌ Critical issues found! Please review and fix.", "RED")
            return False
        else:
            self.print_colored("\n✅ All dependency checks passed!", "GREEN")
            return True


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check dependencies for MetaTrader Python Framework"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory"
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only without detailed output"
    )

    args = parser.parse_args()

    # Initialize checker
    checker = DependencyChecker(args.project_root)

    if args.report_only:
        checker.generate_dependency_report()
    else:
        success = checker.run_all_checks()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()