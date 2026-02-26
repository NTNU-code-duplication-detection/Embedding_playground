"""
Config module for decompiling
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class DecompilerConfig:
    """
    Decompile config
    """
    jdk_home: str
    vineflower_jar: str
