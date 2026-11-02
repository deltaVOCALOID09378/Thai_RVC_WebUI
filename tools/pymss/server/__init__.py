# ==============================================================================
# File: __init__.py
# Path: tools\pymss\server\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from .config import ServerConfig


def create_app(config):
    """Create the FastAPI application.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.

    Returns:
        FastAPI: Configured application instance.

    Example:
        >>> app = create_app()"""
    from .app import create_app as _create_app

    return _create_app(config)


def run_server(config):
    """Run the pymss HTTP server.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.

    Returns:
        None: Runs until the server stops.

    Example:
        >>> run_server()"""
    from .app import run_server as _run_server

    return _run_server(config)


__all__ = ("ServerConfig", "create_app", "run_server")
