"""Russian offline conversation generator."""
__version__ = "2.2.0"


def __getattr__(name):
    # Allow the training and data-preparation modules to start before weights exist.
    if name == "Session":
        from .session import Session
        return Session
    raise AttributeError(name)
