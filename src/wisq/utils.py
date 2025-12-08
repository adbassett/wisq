import os
import shutil
from time import time_ns
from datetime import datetime
import random


# Default directory name for intermediates (created in current working directory)
DEFAULT_INTERMEDIATES_DIR = "wisq_intermediates"

# Mode-specific intermediate filenames
INTERMEDIATE_FILENAMES = {
    "full_ft": "decomposed_circuit.qasm",  # Decomposed circuit before M&R
    "opt": "optimized_circuit.qasm",        # Optimized circuit output
    "scmr": "input_circuit.qasm",           # Input circuit for M&R
}
INTERMEDIATE_RESULT_FILENAMES = {
    "full_ft": "mr_result.json",             # M&R result from full_ft
    "opt": None,                              # No secondary output for opt
    "scmr": "mr_result.json",                 # M&R result from scmr
}
INTERMEDIATE_METADATA_FILENAME = "intermediate_info.txt"

# Backward compatibility
INTERMEDIATE_FILENAME = "decomposed_circuit.qasm"


def create_scratch_dir(output_path: str) -> str:
    """Create temporary scratch directory for GUOQ processing.
    
    Args:
        output_path: Path to the output file (directory will be created alongside it).
        
    Returns:
        Tuple of (scratch_dir_path, unique_id).
    """
    timestamp = time_ns()
    uid = f"{timestamp}_{random.randint(0, 10000000)}"
    scratch_dir_name = f"wisq_tmp_{uid}"
    scratch_dir_path = os.path.join(os.path.dirname(output_path), scratch_dir_name)
    os.mkdir(scratch_dir_path)
    return (scratch_dir_path, uid)


def normalize_intermediates_path(intermediates_dir: str) -> str:
    """Normalize and validate the intermediates directory path.
    
    Converts relative paths to absolute, expands ~ to home directory,
    and ensures the path is valid and writable.
    
    Args:
        intermediates_dir: User-provided directory path (can be relative, absolute, or use ~).
        
    Returns:
        Normalized absolute path to the intermediates directory.
        
    Raises:
        ValueError: If the path is invalid or points to a file instead of directory.
        PermissionError: If the directory cannot be created due to permissions.
    """
    if intermediates_dir is None:
        return None
    
    # Expand ~ to home directory
    expanded_path = os.path.expanduser(intermediates_dir)
    
    # Convert to absolute path
    abs_path = os.path.abspath(expanded_path)
    
    # Check if path exists and is a file (not a directory)
    if os.path.exists(abs_path) and os.path.isfile(abs_path):
        raise ValueError(
            f"Error: '{abs_path}' is a file, not a directory. "
            f"Please provide a directory path for --save_intermediates."
        )
    
    return abs_path


def get_intermediate_filepath(intermediates_dir: str, mode: str = "full_ft") -> str:
    """Get the full path to the saved intermediate circuit file.
    
    Args:
        intermediates_dir: Directory where intermediates are stored.
        mode: Compiler mode ('full_ft', 'opt', 'scmr').
        
    Returns:
        Full path to the intermediate circuit file for the given mode.
    """
    if intermediates_dir is None:
        return None
    filename = INTERMEDIATE_FILENAMES.get(mode, INTERMEDIATE_FILENAME)
    return os.path.join(intermediates_dir, filename)


def get_result_filepath(intermediates_dir: str, mode: str = "full_ft") -> str:
    """Get the full path to the saved result file (e.g., M&R JSON output).
    
    Args:
        intermediates_dir: Directory where intermediates are stored.
        mode: Compiler mode ('full_ft', 'opt', 'scmr').
        
    Returns:
        Full path to the result file, or None if mode doesn't have one.
    """
    if intermediates_dir is None:
        return None
    filename = INTERMEDIATE_RESULT_FILENAMES.get(mode)
    if filename is None:
        return None
    return os.path.join(intermediates_dir, filename)


def get_metadata_filepath(intermediates_dir: str) -> str:
    """Get the full path to the intermediate metadata file.
    
    Args:
        intermediates_dir: Directory where intermediates are stored.
        
    Returns:
        Full path to the intermediate_info.txt file.
    """
    if intermediates_dir is None:
        return None
    return os.path.join(intermediates_dir, INTERMEDIATE_METADATA_FILENAME)


def has_intermediate_file(intermediates_dir: str, mode: str = "full_ft") -> bool:
    """Check if a valid intermediate circuit file exists for the given mode.
    
    Args:
        intermediates_dir: Directory to check for intermediate files.
        mode: Compiler mode ('full_ft', 'opt', 'scmr').
        
    Returns:
        True if a valid intermediate file exists, False otherwise.
    """
    if intermediates_dir is None:
        return False
    
    filepath = get_intermediate_filepath(intermediates_dir, mode)
    if not os.path.exists(filepath):
        return False
    
    # Verify the file is not empty and looks like valid content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(200)  # Read first 200 chars to check validity
            # For QASM files
            if filepath.endswith('.qasm'):
                if 'OPENQASM' in content or 'qreg' in content or 'qubit' in content.lower():
                    return True
                else:
                    print(f"Warning: Found intermediate file but it doesn't appear to be valid QASM. Will regenerate.")
                    return False
            # For JSON files
            elif filepath.endswith('.json'):
                if content.strip().startswith('{') or content.strip().startswith('['):
                    return True
                else:
                    print(f"Warning: Found intermediate file but it doesn't appear to be valid JSON. Will regenerate.")
                    return False
            else:
                # Unknown format, assume valid if not empty
                return len(content.strip()) > 0
    except (IOError, OSError) as e:
        print(f"Warning: Could not read intermediate file: {e}. Will regenerate.")
        return False


def has_result_file(intermediates_dir: str, mode: str = "full_ft") -> bool:
    """Check if a valid result file exists for the given mode.
    
    Args:
        intermediates_dir: Directory to check for result files.
        mode: Compiler mode ('full_ft', 'opt', 'scmr').
        
    Returns:
        True if a valid result file exists, False otherwise.
    """
    if intermediates_dir is None:
        return False
    
    filepath = get_result_filepath(intermediates_dir, mode)
    if filepath is None or not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(100)
            if content.strip().startswith('{') or content.strip().startswith('['):
                return True
    except (IOError, OSError):
        pass
    return False


def save_intermediate_file(
    source_path: str, 
    intermediates_dir: str, 
    input_circuit: str = None,
    mode: str = "full_ft",
    is_result: bool = False
) -> str:
    """Save an intermediate or result file to the intermediates directory.
    
    Creates the directory if it doesn't exist, copies the file,
    and writes metadata about when/how it was created.
    
    Args:
        source_path: Path to the source file to save.
        intermediates_dir: Directory to save the file to.
        input_circuit: Optional path to the original input circuit (for metadata).
        mode: Compiler mode ('full_ft', 'opt', 'scmr').
        is_result: If True, save as result file; if False, save as intermediate.
        
    Returns:
        The absolute path where the file was saved, or None if skipped.
        
    Raises:
        FileNotFoundError: If the source file doesn't exist.
        PermissionError: If the directory cannot be created or written to.
    """
    if intermediates_dir is None:
        return None
    
    # Normalize the path
    abs_dir = normalize_intermediates_path(intermediates_dir)
    
    # Create directory with full path
    try:
        os.makedirs(abs_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Cannot create intermediates directory '{abs_dir}'. "
            f"Check that you have write permissions to this location."
        )
    
    # Verify source file exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Cannot save intermediate: source file '{source_path}' does not exist."
        )
    
    # Determine destination path based on mode and type
    if is_result:
        dest_path = get_result_filepath(abs_dir, mode)
        if dest_path is None:
            return None  # This mode doesn't have a result file
    else:
        dest_path = get_intermediate_filepath(abs_dir, mode)
    
    try:
        shutil.copy2(source_path, dest_path)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to save intermediate file: {e}")
    
    # Write/update metadata file with creation info
    metadata_path = get_metadata_filepath(abs_dir)
    try:
        # Read existing metadata if present
        existing_content = ""
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"WISQ Intermediate Files Metadata\n")
            f.write(f"=================================\n\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if input_circuit:
                f.write(f"Original input: {os.path.abspath(input_circuit)}\n")
            f.write(f"Directory: {abs_dir}\n")
            f.write(f"\n--- Files in this cache ---\n")
            
            # List all files in the directory
            for filename in sorted(os.listdir(abs_dir)):
                if filename == INTERMEDIATE_METADATA_FILENAME:
                    continue
                filepath = os.path.join(abs_dir, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    f.write(f"\n  {filename}:\n")
                    f.write(f"    Size: {stat.st_size} bytes\n")
                    f.write(f"    Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Count gates for QASM files
                    if filename.endswith('.qasm'):
                        with open(filepath, 'r', encoding='utf-8') as qasm:
                            lines = qasm.readlines()
                            gate_count = sum(1 for line in lines if any(
                                gate in line for gate in ['cx ', 't ', 'tdg ', 'h ', 's ', 'sdg ', 'x ', 'y ', 'z ']
                            ))
                            f.write(f"    Approximate gate count: {gate_count}\n")
            
            f.write(f"\n--- Usage ---\n")
            f.write(f"This directory contains cached compilation results.\n")
            f.write(f"Delete this directory or specific files to force re-computation.\n")
            f.write(f"\nSupported modes:\n")
            f.write(f"  - full_ft: Caches decomposed circuit (skip decomposition on rerun)\n")
            f.write(f"  - opt: Caches optimized circuit output\n")
            f.write(f"  - scmr: Caches input circuit and M&R results\n")
    except (IOError, OSError):
        # Metadata is optional, don't fail if we can't write it
        pass
    
    return dest_path


def get_default_intermediates_dir() -> str:
    """Get the default intermediates directory path.
    
    Returns a path in the current working directory that is visible and not hidden.
    
    Returns:
        Absolute path to the default intermediates directory.
    """
    return os.path.abspath(DEFAULT_INTERMEDIATES_DIR)


def print_intermediate_status(
    intermediates_dir: str, 
    action: str = "saved",
    mode: str = "full_ft",
    file_type: str = "intermediate"
) -> None:
    """Print a clear status message about intermediate file operations.
    
    Args:
        intermediates_dir: The directory where intermediates are stored.
        action: The action taken ("saved", "loaded", "skipped").
        mode: The compiler mode ('full_ft', 'opt', 'scmr').
        file_type: Type of file ("intermediate" or "result").
    """
    abs_path = os.path.abspath(intermediates_dir)
    
    if file_type == "result":
        filepath = get_result_filepath(abs_path, mode)
        filename = INTERMEDIATE_RESULT_FILENAMES.get(mode, "result.json")
    else:
        filepath = get_intermediate_filepath(abs_path, mode)
        filename = INTERMEDIATE_FILENAMES.get(mode, INTERMEDIATE_FILENAME)
    
    # Mode-specific descriptions
    mode_descriptions = {
        "full_ft": "decomposed circuit",
        "opt": "optimized circuit",
        "scmr": "input circuit",
    }
    file_desc = mode_descriptions.get(mode, "intermediate file")
    if file_type == "result":
        file_desc = "M&R result"
    
    print(f"\n{'='*60}")
    if action == "saved":
        print(f"✓ [{mode.upper()}] {file_desc.title()} SAVED!")
    elif action == "loaded":
        print(f"✓ [{mode.upper()}] {file_desc.title()} LOADED from cache")
    elif action == "skipped":
        print(f"○ [{mode.upper()}] Skipping - using cached {file_desc}")
    else:
        print(f"  [{mode.upper()}] {action}")
    
    print(f"  Directory: {abs_path}")
    print(f"  File: {filename}")
    
    if filepath and os.path.exists(filepath):
        stat = os.stat(filepath)
        print(f"  Size: {stat.st_size} bytes")
    
    print(f"{'='*60}\n")


def copy_result_to_output(intermediates_dir: str, output_path: str, mode: str = "full_ft") -> bool:
    """Copy cached result file to the expected output location.
    
    Args:
        intermediates_dir: Directory containing cached results.
        output_path: Destination output path.
        mode: Compiler mode.
        
    Returns:
        True if successfully copied, False otherwise.
    """
    result_path = get_result_filepath(intermediates_dir, mode)
    if result_path and os.path.exists(result_path):
        try:
            shutil.copy2(result_path, output_path)
            return True
        except (IOError, OSError):
            pass
    return False
