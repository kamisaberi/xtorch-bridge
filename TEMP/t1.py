import sys
import os

# Add the current directory to the path to ensure it's found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("--- Attempting FINAL direct import ---", flush=True)

try:
    # Since the .so file is in the same directory, this should work
    import xtorch_bridge_impl

    print("\n\n****************************************")
    print("    IT WORKED. THE IMPORT SUCCEEDED.")
    print("****************************************")

    # Now let's see if we can use it
    print("\nAttempting to access a function (this will fail if the module is empty)...")
    # Note: we need to access the 'fit' function from the raw module
    # It won't be in a package namespace.
    print(xtorch_bridge_impl.fit)
    print("\n--- TEST COMPLETE: FULL SUCCESS ---")

except ImportError as e:
    print("\n\n****************************************")
    print("    IT FAILED. FINAL DIAGNOSIS.")
    print("****************************************")
    print("This means the .so file is fundamentally unloadable by Python,")
    print("even when all packaging and path issues are removed.")
    print("The cause is a deep toolchain/system incompatibility.")
    print("-" * 40)
    print(f"The Final Error: {e}")
    print("-" * 40)

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")