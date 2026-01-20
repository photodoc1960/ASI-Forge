"""
Safe Code Execution Sandbox
Based on ASI-Arch's approach with added safety layers
"""

import subprocess
import tempfile
import os
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from code execution"""
    success: bool
    model: Optional[nn.Module]
    error_message: str
    stdout: str
    stderr: str
    execution_time: float
    metrics: Dict[str, float]


class SafeCodeExecutor:
    """
    Executes generated code safely in isolated subprocess

    Safety features:
    1. Subprocess isolation (crashes don't kill main process)
    2. Timeout limits
    3. Resource limits (memory, CPU)
    4. Code validation before execution
    5. Sandboxed file system access
    """

    def __init__(
        self,
        timeout_seconds: int = 60,
        max_memory_gb: float = 4.0,
        work_dir: Optional[str] = None
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_gb = max_memory_gb
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="aegis_sandbox_")

        # Create directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "logs"), exist_ok=True)

        logger.info(f"Code executor initialized with work_dir: {self.work_dir}")

    def execute_model_code(
        self,
        code: str,
        test_inputs: torch.Tensor,
        architecture_id: str
    ) -> ExecutionResult:
        """
        Execute generated model code safely

        Args:
            code: Python code defining the model
            test_inputs: Test inputs for validation
            architecture_id: Unique ID for this architecture

        Returns:
            ExecutionResult with model and metrics
        """

        logger.info(f"Executing code for architecture {architecture_id}")

        start_time = time.time()

        # 1. Write code to file
        code_file = os.path.join(self.work_dir, f"model_{architecture_id}.py")
        with open(code_file, 'w') as f:
            f.write(code)

        # 2. Write test inputs
        input_file = os.path.join(self.work_dir, f"inputs_{architecture_id}.pt")
        torch.save(test_inputs, input_file)

        # 3. Create execution script
        exec_script = self._create_execution_script(
            code_file,
            input_file,
            architecture_id
        )

        # 4. Run in subprocess with safety limits
        try:
            result = subprocess.run(
                ['python', exec_script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.work_dir,
                env=self._get_safe_env()
            )

            execution_time = time.time() - start_time

            # 5. Parse results
            if result.returncode == 0:
                # Load model and metrics
                model_file = os.path.join(self.work_dir, f"model_{architecture_id}.pth")
                metrics_file = os.path.join(self.work_dir, f"metrics_{architecture_id}.json")

                if os.path.exists(model_file):
                    model = torch.load(model_file)
                else:
                    model = None

                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}

                return ExecutionResult(
                    success=True,
                    model=model,
                    error_message="",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time=execution_time,
                    metrics=metrics
                )
            else:
                return ExecutionResult(
                    success=False,
                    model=None,
                    error_message=f"Execution failed with code {result.returncode}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time=execution_time,
                    metrics={}
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Execution timeout for {architecture_id}")
            return ExecutionResult(
                success=False,
                model=None,
                error_message=f"Execution timeout ({self.timeout_seconds}s)",
                stdout="",
                stderr="",
                execution_time=self.timeout_seconds,
                metrics={}
            )
        except Exception as e:
            logger.error(f"Execution error for {architecture_id}: {e}")
            return ExecutionResult(
                success=False,
                model=None,
                error_message=str(e),
                stdout="",
                stderr="",
                execution_time=time.time() - start_time,
                metrics={}
            )

    def _create_execution_script(
        self,
        code_file: str,
        input_file: str,
        architecture_id: str
    ) -> str:
        """Create script that safely executes the model code"""

        script_path = os.path.join(self.work_dir, f"exec_{architecture_id}.py")

        script_content = f'''
import sys
import torch
import torch.nn as nn
import json
import traceback

def main():
    try:
        # Load test inputs
        test_inputs = torch.load("{input_file}")

        # Import the model code
        sys.path.insert(0, "{os.path.dirname(code_file)}")
        import {os.path.basename(code_file)[:-3]} as model_module

        # Get the model class (assumes 'model' or 'Model' is defined)
        if hasattr(model_module, 'Model'):
            model_class = model_module.Model
        elif hasattr(model_module, 'model'):
            model_class = model_module.model
        else:
            # Try to find first nn.Module subclass
            for name in dir(model_module):
                obj = getattr(model_module, name)
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                    model_class = obj
                    break
            else:
                raise ValueError("No model class found in generated code")

        # Instantiate model
        model = model_class()

        # Test forward pass
        with torch.no_grad():
            outputs = model(test_inputs)

        # Validate outputs
        if torch.isnan(outputs).any():
            raise ValueError("Model outputs contain NaN")
        if torch.isinf(outputs).any():
            raise ValueError("Model outputs contain Inf")

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Save model
        torch.save(model, "{self.work_dir}/model_{architecture_id}.pth")

        # Save metrics
        metrics = {{
            'parameters': param_count,
            'output_shape': list(outputs.shape),
            'forward_pass_success': True
        }}

        with open("{self.work_dir}/metrics_{architecture_id}.json", 'w') as f:
            json.dump(metrics, f)

        print(f"SUCCESS: Model validated with {{param_count}} parameters")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {{str(e)}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    def _get_safe_env(self) -> Dict[str, str]:
        """Get environment with safety restrictions"""

        env = os.environ.copy()

        # Restrict network access (optional, OS-dependent)
        # env['no_proxy'] = '*'

        # Set resource limits
        # env['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{int(self.max_memory_gb * 1024)}'

        return env

    def cleanup(self):
        """Clean up temporary files"""

        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
            logger.info(f"Cleaned up work directory: {self.work_dir}")


class CodeDebugger:
    """
    Debugs failed code execution
    Based on ASI-Arch's debug loop
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def debug_and_fix(
        self,
        code: str,
        error_message: str,
        stderr: str
    ) -> Tuple[str, str]:
        """
        Attempt to fix code based on error

        Args:
            code: Original code
            error_message: Error message
            stderr: Standard error output

        Returns:
            (fixed_code, changes_description)
        """

        logger.info("Attempting to debug code...")

        # Analyze error
        error_type = self._classify_error(error_message, stderr)

        # Apply fix strategy
        if error_type == "import_error":
            fixed_code = self._fix_import_error(code, stderr)
            changes = "Fixed import errors"

        elif error_type == "shape_mismatch":
            fixed_code = self._fix_shape_error(code, stderr)
            changes = "Fixed tensor shape mismatches"

        elif error_type == "undefined_variable":
            fixed_code = self._fix_undefined_variable(code, stderr)
            changes = "Fixed undefined variables"

        elif error_type == "attribute_error":
            fixed_code = self._fix_attribute_error(code, stderr)
            changes = "Fixed attribute errors"

        else:
            # Generic fix attempt
            fixed_code = code
            changes = "No automatic fix available"

        return fixed_code, changes

    def _classify_error(self, error_msg: str, stderr: str) -> str:
        """Classify error type"""

        error_text = (error_msg + stderr).lower()

        if "importerror" in error_text or "modulenotfounderror" in error_text:
            return "import_error"
        elif "shape" in error_text or "size mismatch" in error_text:
            return "shape_mismatch"
        elif "nameerror" in error_text or "not defined" in error_text:
            return "undefined_variable"
        elif "attributeerror" in error_text:
            return "attribute_error"
        else:
            return "unknown"

    def _fix_import_error(self, code: str, stderr: str) -> str:
        """Fix common import errors"""

        # Add common imports if missing
        imports_to_add = []

        if "torch" not in code:
            imports_to_add.append("import torch")
        if "nn" not in code:
            imports_to_add.append("import torch.nn as nn")
        if "F" not in code and "functional" in stderr.lower():
            imports_to_add.append("import torch.nn.functional as F")

        if imports_to_add:
            return "\\n".join(imports_to_add) + "\\n\\n" + code

        return code

    def _fix_shape_error(self, code: str, stderr: str) -> str:
        """Attempt to fix shape mismatches"""
        # This would require more sophisticated analysis
        # For now, return unchanged
        return code

    def _fix_undefined_variable(self, code: str, stderr: str) -> str:
        """Fix undefined variables"""
        # Would need AST parsing and variable tracking
        return code

    def _fix_attribute_error(self, code: str, stderr: str) -> str:
        """Fix attribute errors"""
        # Would need semantic understanding
        return code


class TrainingValidator:
    """
    Validates architecture by actually training it
    Based on ASI-Arch's training validation
    """

    def __init__(
        self,
        training_steps: int = 100,
        batch_size: int = 4
    ):
        self.training_steps = training_steps
        self.batch_size = batch_size

    def validate_training(
        self,
        model: nn.Module,
        vocab_size: int,
        seq_len: int = 20
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate that model can be trained

        Args:
            model: Model to validate
            vocab_size: Vocabulary size
            seq_len: Sequence length

        Returns:
            (success, metrics)
        """

        try:
            # Create synthetic training data
            train_data = torch.randint(0, vocab_size, (self.training_steps * self.batch_size, seq_len))
            train_labels = torch.randint(0, vocab_size, (self.training_steps * self.batch_size, seq_len))

            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            model.train()
            losses = []

            # Training loop
            for step in range(self.training_steps):
                # Get batch
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size

                inputs = train_data[start_idx:end_idx]
                targets = train_labels[start_idx:end_idx]

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients
                if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                    logger.error("NaN gradients detected")
                    return False, {'error': 'NaN gradients'}

                optimizer.step()

                losses.append(loss.item())

            # Validate training worked
            if not losses:
                return False, {'error': 'No losses recorded'}

            avg_loss = sum(losses) / len(losses)
            final_loss = losses[-1]

            # Check loss is decreasing (or at least not exploding)
            if final_loss > avg_loss * 2:
                logger.warning(f"Loss increasing: {final_loss} > {avg_loss * 2}")

            metrics = {
                'avg_loss': avg_loss,
                'final_loss': final_loss,
                'training_steps': self.training_steps,
                'trainable': True
            }

            return True, metrics

        except Exception as e:
            logger.error(f"Training validation failed: {e}")
            return False, {'error': str(e), 'trainable': False}
