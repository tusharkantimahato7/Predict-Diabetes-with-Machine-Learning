import os

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def print_separator(title=""):
    """Print a visual separator for console output."""
    print("\n" + "="*50)
    if title:
        print(f"  {title}")
        print("="*50)