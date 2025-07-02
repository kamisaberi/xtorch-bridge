print("--- Attempting to import the local .so file ---")

try:
    # Because the .so file is in the same directory, Python can find it.
    # We import it by its filename (without the long .cpython...so part).
    import xtorch_bridge_impl

    print("\nSUCCESS! The C++ module was imported successfully.")

    # Let's see if we can access the functions you defined
    print("Accessing functions from the module...")
    print(f"  Found: {xtorch_bridge_impl.fit}")
    print("\n--- TEST COMPLETE ---")

except ImportError as e:
    print(f"\nIMPORT FAILED. Error: {e}")
    print("This can happen if the RPATH is not set correctly in the .so file.")