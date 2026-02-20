from termcolor import colored

def info(msg: str) -> None:
    """Print info message in blue."""
    print(colored("[INFO]", "blue"), f"{msg}")

def success(msg: str) -> None:
    """Print success message in green."""
    print(colored("[SUCCESS]", "green"), f"{msg}")

def error(msg: str) -> None:
    """Print error message in red."""
    print(colored("[ERROR]", "red"), f"{msg}")

def warning(msg: str) -> None:
    """Print warning message in yellow."""
    print(colored("[WARNING]", "yellow"), f"{msg}")