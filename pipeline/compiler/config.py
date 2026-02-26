"""
Config module
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class CompilerConfig:
    """
    Compiler config for jdk home path
    """
    jdk_home: str
