# =====================================================================
# IMPORTS
# =====================================================================
import os
import joblib
import time
import tempfile
import glob

# =====================================================================
# UTILITY OPERATIONS (File IO)
# =====================================================================
def save_something(data, filename, directory="/results"):
    """
    Safely saves a python object
    to a specified scratch directory, creating the directory if it doesn't exist.
    """
    # Create the folder if it doesn't exist yet
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Build the full file path
    filepath = os.path.join(directory, f"{filename}.pkl")
    
    # Save using joblib (excellent for compressing dictionaries containing numpy arrays)
    joblib.dump(data, filepath)
    print(f"Successfully saved results to: {filepath}")

def load_something(filename, directory="/results"):
    """
    Loads a saved results file back into memory.
    """
    filepath = os.path.join(directory, f"{filename}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved file found at {filepath}")
        
    return joblib.load(filepath)

# =====================================================================
# UTILITY OPERATIONS (File IO)
# =====================================================================

def save_optimization_result(res_dict, base_name, prefix_dir="results_data", make_new_dir=True):
    """
    Safely saves an optimization dictionary to disk using timestamps to avoid overwrites.
    Guarantees atomic execution using temporary files.
    """

    # 2. Generate a clean timestamp suffix (YYYYMMDD_HHMMSS)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{base_name}_{timestamp}"
    filename = name + ".pkl"
    if make_new_dir:
        dir_filepath = os.path.join(prefix_dir, name)
    else:
        dir_filepath = prefix_dir
    final_filepath = os.path.join(dir_filepath, filename)

    # 1. Ensure the directory exists safely
    try:
        os.makedirs(dir_filepath, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {prefix_dir} due to {e}. Saving to current directory.")
        prefix_dir = "."

    # 3. ATOMIC WRITE: Save to a temporary file first to prevent corruption
    try:
        with tempfile.NamedTemporaryFile('wb', dir=prefix_dir, delete=False) as tf:
            joblib.dump(res_dict, tf, compress=3)  # Compresses numpy structures nicely
            temp_output_path = tf.name
        
        # Atomically swap the temporary pointer for the real final filepath
        os.replace(temp_output_path, final_filepath)
        print(f"Successfully stored optimization data at: {final_filepath}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save data! Details: {e}")
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise e