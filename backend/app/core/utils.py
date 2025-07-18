# backend/app/core/utils.py

import os
from pathlib import Path
import threading

# Assuming settings and PROJECT_ROOT_DIR are correctly defined in config.py
# If not, you might need to adjust how COUNTER_FILE_PATH is determined.
# For this example, let's assume config.py can be imported to get PROJECT_ROOT_DIR
try:
    from .config import PROJECT_ROOT_DIR # Relative import if utils.py is in core
except ImportError:
    # Fallback if running script directly or structure is different
    # This path assumes utils.py is in backend/app/core/
    # and data directory is at project_root/data/
    current_file_dir = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = current_file_dir.parents[2] # core -> app -> backend -> project_root

DATA_DIR = PROJECT_ROOT_DIR / "data"
COUNTER_FILE_PATH = DATA_DIR / "doc_id_counter.txt"
ID_PREFIX = "DOC_"
ID_PADDING = 4  # For DOC_0001, DOC_0010, DOC_0100, DOC_1000 etc.

# Thread lock to make counter increment thread-safe
# Important if multiple uploads could happen concurrently and try to access the counter file.
# FastAPI background tasks can run concurrently.
_counter_lock = threading.Lock()

def get_next_document_id() -> str:
    """
    Generates the next sequential document ID (e.g., DOC_0001).
    Reads the last used number from a counter file, increments it,
    saves it back, and returns the formatted ID.
    This function is made thread-safe.
    """
    with _counter_lock:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        current_count = 0
        try:
            if COUNTER_FILE_PATH.exists():
                with open(COUNTER_FILE_PATH, "r") as f:
                    content = f.read().strip()
                    if content.isdigit():
                        current_count = int(content)
        except Exception as e:
            print(f"Warning: Could not read counter file {COUNTER_FILE_PATH}. Starting from 0. Error: {e}")
            current_count = 0 # Reset on error

        next_count = current_count + 1

        try:
            with open(COUNTER_FILE_PATH, "w") as f:
                f.write(str(next_count))
        except Exception as e:
            print(f"Error: Could not write to counter file {COUNTER_FILE_PATH}. ID sequence might not persist. Error: {e}")
            # Decide if you want to raise an error or just proceed with an in-memory increment for this session

        # Format the ID string, e.g., DOC_0001
        formatted_id = f"{ID_PREFIX}{str(next_count).zfill(ID_PADDING)}"
        return formatted_id

if __name__ == "__main__":
    # Test the counter
    print("Testing document ID generation:")
    # Ensure the counter file is reset for a clean test run if needed
    if COUNTER_FILE_PATH.exists():
        os.remove(COUNTER_FILE_PATH)
        print(f"Removed existing counter file for test: {COUNTER_FILE_PATH}")

    ids_generated = []
    for i in range(5):
        doc_id = get_next_document_id()
        ids_generated.append(doc_id)
        print(f"Generated ID {i+1}: {doc_id}")

    print("\nVerifying counter file content...")
    if COUNTER_FILE_PATH.exists():
        with open(COUNTER_FILE_PATH, "r") as f:
            final_count = f.read().strip()
            print(f"Counter file content: {final_count}")
            expected_count = str(len(ids_generated))
            if final_count == expected_count:
                print(f"Counter file correctly updated to {expected_count}.")
            else:
                print(f"Error: Counter file shows {final_count}, expected {expected_count}.")
    else:
        print(f"Error: Counter file {COUNTER_FILE_PATH} was not created.")

    print("\nTesting thread safety (simulated)...")
    # This is a very basic simulation. Real thread safety testing is more complex.
    if COUNTER_FILE_PATH.exists(): # Reset for this test
        os.remove(COUNTER_FILE_PATH)
    
    results = []
    def generate_ids_threaded(num_ids):
        for _ in range(num_ids):
            results.append(get_next_document_id())

    thread1 = threading.Thread(target=generate_ids_threaded, args=(3,))
    thread2 = threading.Thread(target=generate_ids_threaded, args=(3,))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print(f"Generated IDs in threaded test (should be unique and sequential): {sorted(results)}")
    print(f"Total unique IDs generated: {len(set(results))}")
    if COUNTER_FILE_PATH.exists():
        with open(COUNTER_FILE_PATH, "r") as f:
            final_count_threaded = f.read().strip()
            print(f"Counter file content after threaded test: {final_count_threaded}")
            if len(set(results)) == 6 and final_count_threaded == "6":
                 print("Threaded test suggests counter is working as expected.")
            else:
                print("Potential issue in threaded counter update.")
