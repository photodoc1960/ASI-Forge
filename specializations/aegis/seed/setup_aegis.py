"""
AEGIS Setup and Installation Script
Automatically detects hardware and configures the system
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.auto_configure import auto_setup, HardwareDetector, AutoConfigurator


def print_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                    AEGIS SETUP & INSTALLATION                        ║
║         Adaptive Evolutionary General Intelligence System           ║
║                                                                      ║
║  This script will:                                                   ║
║  1. Detect your hardware configuration                               ║
║  2. Check minimum system requirements                                ║
║  3. Optimize AEGIS for your system                                   ║
║  4. Create configuration file                                        ║
║  5. Verify installation                                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def verify_dependencies():
    """Verify required dependencies are installed"""
    print("\n📦 Verifying dependencies...")

    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'psutil': 'psutil'
    }

    missing = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

    print("✓ All dependencies installed\n")
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    dirs = [
        'logs',
        'checkpoints',
        'knowledge_base',
        'exports'
    ]

    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")

    print("✓ Directories created\n")


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running verification tests...")

    try:
        import torch

        # Test PyTorch
        x = torch.randn(5, 5)
        y = x.mean()
        print("  ✓ PyTorch working")

        # Test CUDA if available
        if torch.cuda.is_available():
            x = x.cuda()
            y = x.mean()
            print(f"  ✓ CUDA working ({torch.cuda.device_count()} GPU(s))")
        else:
            print("  ℹ CUDA not available (CPU mode)")

        # Test AEGIS imports
        from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel
        from core.safety.safety_validator import ComprehensiveSafetyValidator
        from core.agency.autonomous_agent import AutonomousAgent
        print("  ✓ AEGIS modules loading")

        print("✓ All tests passed\n")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def generate_quick_start_guide(config):
    """Generate a quick start guide"""

    guide = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         QUICK START GUIDE                            ║
╚══════════════════════════════════════════════════════════════════════╝

Your AEGIS installation is configured for:
  • Device: {config.device.upper()}
  • Model: {config.d_model}-dimensional with {config.high_level_layers}+{config.low_level_layers} layers
  • Batch size: {config.batch_size}
  • Population: {config.population_size} architectures

OPTION 1: Run Interactive Demo
────────────────────────────────
  python demo.py

OPTION 2: Start Interactive Session
────────────────────────────────────
  python aegis_autonomous.py

OPTION 3: Use Python API
─────────────────────────
  from aegis_autonomous import AutonomousAEGIS, AEGISConfig
  from core.auto_configure import AutoConfigurator

  # Load optimized config
  config = AutoConfigurator.load_config("aegis_config.json")

  # Create AEGIS
  aegis = AutonomousAEGIS(config)

  # Start interactive session
  aegis.interactive_session()

  # Or run autonomous operation
  aegis.start_autonomous_operation(max_iterations=100)

IMPORTANT NOTES:
────────────────
  • All architecture changes require your approval
  • System will automatically freeze if anomalies detected
  • You can emergency stop at any time: aegis.emergency_stop("reason")
  • Configuration file: aegis_config.json

DOCUMENTATION:
──────────────
  • README.md - Overview and introduction
  • SETUP.md - Detailed setup instructions
  • EXAMPLES.md - Usage examples
  • SAFETY.md - Safety features and protocols

For help, visit: https://github.com/your-repo/aegis/issues

╚══════════════════════════════════════════════════════════════════════╝
    """

    # Save to file
    with open("QUICKSTART.txt", "w") as f:
        f.write(guide)

    print(guide)


def main():
    """Main setup process"""

    print_banner()

    # Verify dependencies
    if not verify_dependencies():
        sys.exit(1)

    # Create directories
    create_directories()

    # Auto-configure
    try:
        config = auto_setup()
    except Exception as e:
        print(f"\n❌ Auto-configuration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but installation may still work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Generate quick start guide
    generate_quick_start_guide(config)

    # Save guide
    print("\n✓ Quick start guide saved to: QUICKSTART.txt")

    print("\n" + "="*70)
    print("🎉 AEGIS INSTALLATION COMPLETE! 🎉")
    print("="*70)
    print("\nYou can now start using AEGIS!")
    print("\nRecommended first step:")
    print("  python demo.py")
    print("\nFor interactive use:")
    print("  python aegis_autonomous.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
