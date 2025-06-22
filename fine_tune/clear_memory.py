import torch
import gc
import psutil
import os


def print_mps_memory():
    """Print MPS and system memory usage"""
    print("=" * 50)

    # MPS status
    if torch.backends.mps.is_available():
        print(f"MPS Available: ✅")
        print(f"MPS Built: {torch.backends.mps.is_built()}")
    else:
        print(f"MPS Available: ❌")

    # System memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    print(f"Process Memory Usage: {memory_info.rss / 1024**3:.2f} GB")
    print(f"Memory Percentage: {memory_percent:.1f}%")

    # System-wide memory
    system_memory = psutil.virtual_memory()
    print(
        f"System Memory - Used: {system_memory.used / 1024**3:.2f} GB / {system_memory.total / 1024**3:.2f} GB"
    )
    print(f"System Memory Available: {system_memory.available / 1024**3:.2f} GB")
    print("=" * 50)


def free_mps_memory():
    """Free MPS and system memory"""
    print("🧹 Cleaning up memory...")

    # Clear MPS cache (equivalent to torch.cuda.empty_cache())
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("✅ MPS cache cleared")
    else:
        print("⚠️  MPS not available")

    # Force garbage collection
    gc.collect()
    print("✅ Garbage collection completed")
    print()


def main():
    """Memory cleanup script - assumes model is already loaded"""

    print("📊 MEMORY STATE BEFORE CLEANUP")
    print_mps_memory()

    print("\n🧹 FREEING MEMORY...")
    free_mps_memory()

    print("📊 MEMORY STATE AFTER CLEANUP")
    print_mps_memory()


if __name__ == "__main__":
    main()
