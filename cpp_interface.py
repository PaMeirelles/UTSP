import platform
import subprocess
import re
from pathlib import Path


def get_solver_params(num_nodes: int) -> dict:
    """Get recommended solver parameters based on problem size."""
    if num_nodes == 100:
        return {
            'rec_only': 0, 'M': 6, 'K': 10, 'alpha': 0.1, 'beta': 30,
            'ph': 3, 'restart': 0, 'restart_rec': 0
        }
    elif num_nodes == 200:
        return {
            'rec_only': 0, 'M': 5, 'K': 30, 'alpha': 0, 'beta': 100,
            'ph': 2, 'restart': 1, 'restart_rec': 1
        }
    elif num_nodes == 500:
        return {
            'rec_only': 0, 'M': 5, 'K': 30, 'alpha': 0, 'beta': 100,
            'ph': 2, 'restart': 1, 'restart_rec': 1
        }
    elif num_nodes == 1000:
        return {
            'rec_only': 0, 'M': 5, 'K': 10, 'alpha': 0, 'beta': 150,
            'ph': 3, 'restart': 1, 'restart_rec': 1
        }
    else:
        return {
            'rec_only': 0, 'M': 5, 'K': 30, 'alpha': 0, 'beta': 100,
            'ph': 2, 'restart': 1, 'restart_rec': 1
        }


def patch_cpp_constants(search_dir: Path, num_nodes: int):
    """
    Automatically finds and updates Batch Size and Total Instance constants
    in the C++ source code to ensure it only runs 1 instance.
    """
    print("Patching C++ source code to run single instance...")

    # Files to inspect (source and headers)
    files_to_check = list(search_dir.rglob('*.cpp')) + list(search_dir.rglob('*.h'))

    # 1. Update standard Max_City_Num (usually in TSP_IO.h)
    io_header = next(search_dir.rglob('TSP_IO.h'), None)
    if io_header:
        with open(io_header, 'r') as f:
            content = f.read()
        content = re.sub(r'#define Max_City_Num\s+\d+', f'#define Max_City_Num {num_nodes}', content)
        content = re.sub(r'int Rec_Num\s*=\s*\d+;', f'int Rec_Num = 20;', content)
        with open(io_header, 'w') as f:
            f.write(content)

    # 2. Detect and Patch Batch/Instance Counts
    # We look for variables that are printed in the log messages you saw
    log_patterns = {
        "Total Instance Considered": r'Total Instance Considered:?"\s*<<\s*([A-Za-z0-9_]+)',
        "Inst Num Per Batch": r'Inst Num Per Batch\s*"?\s*<<\s*([A-Za-z0-9_]+)'
    }

    vars_to_patch = set()

    # Find variable names
    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                text = f.read()
            for key, pattern in log_patterns.items():
                if key in text:
                    match = re.search(pattern, text)
                    if match:
                        var_name = match.group(1)
                        vars_to_patch.add(var_name)
                        print(f"  Found config variable: {var_name}")
        except:
            continue

    # Patch variable definitions to 1
    for filepath in files_to_check:
        updated = False
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            for var_name in vars_to_patch:
                # Patch #define VAR 128
                regex_define = rf'(#define\s+{var_name}\s+)(\d+)'
                if re.search(regex_define, content):
                    content = re.sub(regex_define, f'\\g<1>1', content)
                    updated = True
                    print(f"  Patched {var_name} -> 1 in {filepath.name}")

                # Patch int VAR = 128;
                regex_assign = rf'(\b{var_name}\s*=\s*)(\d+)(\s*;)'
                if re.search(regex_assign, content):
                    content = re.sub(regex_assign, f'\\g<1>1\\g<3>', content)
                    updated = True
                    print(f"  Patched {var_name} -> 1 in {filepath.name}")

            if updated:
                with open(filepath, 'w') as f:
                    f.write(content)
        except Exception as e:
            print(f"Error patching {filepath}: {e}")


def ensure_solver_compiled(num_nodes: int) -> Path:
    """Ensure the C++ solver is compiled and return path to executable."""
    search_dir = Path('Search')
    is_windows = platform.system() == 'Windows'
    exe_name = 'test.exe' if is_windows else 'test'
    executable = search_dir / exe_name

    # Always patch code to ensure settings are correct for this run
    try:
        patch_cpp_constants(search_dir, num_nodes)
    except Exception as e:
        print(f"Warning: Auto-patching failed ({e}). Compilation might rely on defaults.")

    # Re-compile if we suspect changes or if missing
    # (Since we just patched files, we SHOULD recompile. We assume 'make' checks timestamps)
    print(f"Compiling C++ solver...")

    # Locate main source file
    source_files = list(search_dir.rglob('TSP.cpp'))
    if not source_files:
        raise FileNotFoundError(f"Could not find 'TSP.cpp' inside {search_dir}")
    main_source = source_files[0]

    # Locate include dir
    header_files = list(search_dir.rglob('TSP_IO.h'))
    include_dir = header_files[0].parent if header_files else None

    try:
        if is_windows:
            cmd = ['g++', '-O3', '-o', exe_name, str(main_source.relative_to(search_dir))]
            if include_dir:
                cmd.append(f'-I{include_dir.relative_to(search_dir)}')

            result = subprocess.run(cmd, cwd=search_dir, capture_output=True, text=True, timeout=60)
        else:
            result = subprocess.run(['make'], cwd=search_dir, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        print("Compilation successful!")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Compilation timed out")
    except FileNotFoundError:
        raise RuntimeError("Compiler not found. Ensure MinGW (g++) is installed.")

    if not executable.exists():
        raise RuntimeError("Executable not found after compilation.")

    return executable


def run_solver(executable: Path, input_file: Path, output_file: Path,
               num_nodes: int, params: dict, topk: int, timeout: int):
    """Run the C++ solver with specified parameters."""

    if not input_file.exists():
        raise RuntimeError(f"Input file missing: {input_file}")

    cmd = [
        str(executable.resolve()), '0', str(output_file.resolve()), str(input_file.resolve()),
        str(num_nodes), '1', str(params['rec_only']), str(params['M']), str(params['K']),
        str(params['alpha']), str(params['beta']), str(params['ph']),
        str(params['restart']), str(params['restart_rec'])
    ]

    print(f"Running C++ Solver...")

    try:
        # capture_output=False allows you to see the C++ log in real-time
        result = subprocess.run(
            cmd, cwd=executable.parent, capture_output=False, text=True, timeout=timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Solver failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        print("\n!!! TIMEOUT !!!")
        raise RuntimeError(f"Solver exceeded timeout of {timeout} seconds")