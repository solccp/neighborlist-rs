import numpy as np
from typing import List, Optional, Dict, Union, Any, Tuple

class PyCell:
    def __init__(self, h: List[List[float]], pbc: Optional[List[bool]] = None) -> None: ...
    def wrap(self, pos: List[float]) -> List[float]: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def from_ase(atoms: Any) -> 'PyCell': ...

def build_neighborlists(
    cell: Optional[PyCell],
    positions: np.ndarray,
    cutoff: float,
    parallel: bool = True
) -> Dict[str, np.ndarray]: ...

def build_from_ase(
    atoms: Any,
    cutoff: float
) -> Dict[str, np.ndarray]: ...

def build_multi_from_ase(
    atoms: Any,
    cutoffs: List[float],
    labels: Optional[List[str]] = None,
    disjoint: bool = False
) -> Dict[Union[int, str], Dict[str, np.ndarray]]: ...

def build_neighborlists_multi(
    cell: Optional[PyCell],
    positions: np.ndarray,
    cutoffs: List[float],
    labels: Optional[List[str]] = None,
    disjoint: bool = False
) -> Dict[Union[int, str], Dict[str, np.ndarray]]: ...

def build_neighborlists_batch(
    positions: np.ndarray,
    batch: np.ndarray,
    cells: Optional[np.ndarray] = None,
    cutoff: float = 5.0,
    parallel: bool = True
) -> Dict[str, np.ndarray]: ...

def build_neighborlists_batch_multi(
    positions: np.ndarray,
    batch: np.ndarray,
    cells: Optional[np.ndarray] = None,
    cutoffs: List[float] = [5.0],
    labels: Optional[List[str]] = None,
    disjoint: bool = False
) -> Dict[Union[int, str], Dict[str, np.ndarray]]: ...

def get_num_threads() -> int: ...
def set_num_threads(n: int) -> None: ...
def init_logging(level: Optional[str] = None) -> None: ...
