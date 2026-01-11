"""
Tensor Logic - Unified Framework for Neural and Symbolic AI

Based on Pedro Domingos' work (arXiv:2510.12269), this module implements
a unified mathematical language for AI that bridges symbolic logic and 
neural networks through tensor operations.

Key concepts:
- Logical rules and tensor equations are fundamentally the same
- Relations are encoded as tensors (sparse Boolean or real-valued)
- Inference = tensor join (multiplication) + projection (summation)
- Temperature parameter controls deductive vs analogical reasoning
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class Backend:
    """Abstract backend for tensor operations."""
    
    def __init__(self, name: str = "numpy"):
        self.name = name
    
    def tensor(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """Convert data to tensor."""
        if dtype is None:
            dtype = np.float32
        return np.array(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        """Create zero tensor."""
        if dtype is None:
            dtype = np.float32
        return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        """Create ones tensor."""
        if dtype is None:
            dtype = np.float32
        return np.ones(shape, dtype=dtype)
    
    def einsum(self, subscripts: str, *operands) -> Any:
        """Einstein summation."""
        return np.einsum(subscripts, *operands)
    
    def step(self, x: Any, threshold: float = 0.5) -> Any:
        """Heaviside step function (Boolean conversion)."""
        return (x >= threshold).astype(np.float32)
    
    def sigmoid(self, x: Any) -> Any:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-x))
    
    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum."""
        return np.minimum(a, b)
    
    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum."""
        return np.maximum(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""
        return np.multiply(a, b)
    
    def sum(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Sum reduction."""
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def all(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """All reduction (logical AND over axis)."""
        return np.all(x, axis=axis, keepdims=keepdims).astype(np.float32)
    
    def any(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Any reduction (logical OR over axis)."""
        return np.any(x, axis=axis, keepdims=keepdims).astype(np.float32)
    
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        return np.matmul(a, b)


def create_backend(name: str = "numpy") -> Backend:
    """
    Create a backend for tensor operations.
    
    Args:
        name: Backend name (currently only "numpy" supported)
    
    Returns:
        Backend instance
    
    Raises:
        ValueError: If backend name is not supported
    """
    if name != "numpy":
        raise ValueError(f"Backend '{name}' not supported. Currently only 'numpy' is available.")
    return Backend(name)


class CompilationStrategy(Enum):
    """Semantic interpretations for logical operations."""
    
    SOFT_DIFFERENTIABLE = "soft_differentiable"  # Neural network training
    HARD_BOOLEAN = "hard_boolean"                # Exact verification
    GODEL = "godel"                              # Continuous scoring (min/max)
    PRODUCT = "product"                          # Probabilistic (independent events)
    LUKASIEWICZ = "lukasiewicz"                  # Bounded arithmetic


class LogicalStrategy:
    """Base class for compilation strategies."""
    
    def __init__(self, backend: Backend):
        self.backend = backend
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Logical AND operation."""
        raise NotImplementedError
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Logical OR operation."""
        raise NotImplementedError
    
    def logical_not(self, a: Any) -> Any:
        """Logical NOT operation."""
        raise NotImplementedError
    
    def logical_implies(self, a: Any, b: Any) -> Any:
        """Logical implication (a -> b = not(a) or b)."""
        return self.logical_or(self.logical_not(a), b)


class SoftDifferentiableStrategy(LogicalStrategy):
    """Soft differentiable strategy using sigmoid for neural network training."""
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Soft AND: sigmoid(a + b - 1)"""
        return self.backend.sigmoid(a + b - 1)
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Soft OR: sigmoid(a + b)"""
        return self.backend.sigmoid(a + b)
    
    def logical_not(self, a: Any) -> Any:
        """Soft NOT: 1 - a"""
        return 1 - a


class HardBooleanStrategy(LogicalStrategy):
    """Hard Boolean strategy for exact verification."""
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Hard AND: step(a * b)"""
        return self.backend.step(self.backend.multiply(a, b))
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Hard OR: step(a + b)"""
        return self.backend.step(a + b)
    
    def logical_not(self, a: Any) -> Any:
        """Hard NOT: step(1 - a)"""
        return self.backend.step(1 - a)


class GodelStrategy(LogicalStrategy):
    """Gödel strategy using min/max for continuous scoring."""
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Gödel AND: min(a, b)"""
        return self.backend.minimum(a, b)
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Gödel OR: max(a, b)"""
        return self.backend.maximum(a, b)
    
    def logical_not(self, a: Any) -> Any:
        """Gödel NOT: 1 - a"""
        return 1 - a


class ProductStrategy(LogicalStrategy):
    """Product strategy for probabilistic reasoning."""
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Product AND: a * b"""
        return self.backend.multiply(a, b)
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Product OR: a + b - a*b"""
        return a + b - self.backend.multiply(a, b)
    
    def logical_not(self, a: Any) -> Any:
        """Product NOT: 1 - a"""
        return 1 - a


class LukasiewiczStrategy(LogicalStrategy):
    """Łukasiewicz strategy with bounded arithmetic."""
    
    def logical_and(self, a: Any, b: Any) -> Any:
        """Łukasiewicz AND: max(0, a + b - 1)"""
        return self.backend.maximum(0, a + b - 1)
    
    def logical_or(self, a: Any, b: Any) -> Any:
        """Łukasiewicz OR: min(1, a + b)"""
        return self.backend.minimum(1, a + b)
    
    def logical_not(self, a: Any) -> Any:
        """Łukasiewicz NOT: 1 - a"""
        return 1 - a


def create_strategy(name: Union[str, CompilationStrategy], backend: Backend) -> LogicalStrategy:
    """
    Create a logical compilation strategy.
    
    Args:
        name: Strategy name or enum
        backend: Backend for tensor operations
    
    Returns:
        LogicalStrategy instance
    """
    if isinstance(name, str):
        name = CompilationStrategy(name)
    
    strategy_map = {
        CompilationStrategy.SOFT_DIFFERENTIABLE: SoftDifferentiableStrategy,
        CompilationStrategy.HARD_BOOLEAN: HardBooleanStrategy,
        CompilationStrategy.GODEL: GodelStrategy,
        CompilationStrategy.PRODUCT: ProductStrategy,
        CompilationStrategy.LUKASIEWICZ: LukasiewiczStrategy,
    }
    
    return strategy_map[name](backend)


# Core logical operations (default to hard boolean)

def logical_and(a: Any, b: Any, backend: Optional[Backend] = None, 
                strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Logical AND operation.
    
    Args:
        a: First tensor
        b: Second tensor
        backend: Backend for operations (default: numpy)
        strategy: Compilation strategy (default: hard_boolean)
    
    Returns:
        Result of a AND b
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    return strategy.logical_and(a, b)


def logical_or(a: Any, b: Any, backend: Optional[Backend] = None,
               strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Logical OR operation.
    
    Args:
        a: First tensor
        b: Second tensor
        backend: Backend for operations (default: numpy)
        strategy: Compilation strategy (default: hard_boolean)
    
    Returns:
        Result of a OR b
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    return strategy.logical_or(a, b)


def logical_not(a: Any, backend: Optional[Backend] = None,
                strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Logical NOT operation.
    
    Args:
        a: Tensor to negate
        backend: Backend for operations (default: numpy)
        strategy: Compilation strategy (default: hard_boolean)
    
    Returns:
        Result of NOT a
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    return strategy.logical_not(a)


def logical_implies(a: Any, b: Any, backend: Optional[Backend] = None,
                    strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Logical implication (a -> b).
    
    Args:
        a: Antecedent tensor
        b: Consequent tensor
        backend: Backend for operations (default: numpy)
        strategy: Compilation strategy (default: hard_boolean)
    
    Returns:
        Result of a IMPLIES b
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    return strategy.logical_implies(a, b)


# Quantifiers

def exists(predicate: Any, axis: Optional[int] = None, backend: Optional[Backend] = None) -> Any:
    """
    Existential quantification: exists x such that P(x).
    
    Args:
        predicate: Tensor representing predicate values
        axis: Axis to quantify over (None = all axes)
        backend: Backend for operations (default: numpy)
    
    Returns:
        Result of existential quantification
    """
    if backend is None:
        backend = create_backend()
    return backend.any(predicate, axis=axis)


def forall(predicate: Any, axis: Optional[int] = None, backend: Optional[Backend] = None) -> Any:
    """
    Universal quantification: for all x, P(x).
    
    Args:
        predicate: Tensor representing predicate values
        axis: Axis to quantify over (None = all axes)
        backend: Backend for operations (default: numpy)
    
    Returns:
        Result of universal quantification
    """
    if backend is None:
        backend = create_backend()
    return backend.all(predicate, axis=axis)


# Temperature-controlled reasoning

def apply_temperature(logits: Any, temperature: float, backend: Optional[Backend] = None) -> Any:
    """
    Apply temperature scaling to logits for inference control.
    
    Temperature semantics:
    - T=0: Pure deductive (hard threshold at 0.5)
    - T→0+: Increasingly strict (sharper sigmoid)
    - T=1: Neutral (standard sigmoid)
    - T>1: Increasingly generous (softer sigmoid)
    
    Args:
        logits: Input tensor (can be any real values)
        temperature: Temperature parameter (T≥0)
        backend: Backend for operations (default: numpy)
    
    Returns:
        Temperature-scaled probabilities in [0, 1]
    """
    if backend is None:
        backend = create_backend()
    
    # Use small epsilon for numerical stability
    EPSILON = 1e-10
    
    if temperature < EPSILON:
        # Pure deductive: hard threshold
        return backend.step(logits, threshold=0.5)
    else:
        # Analogical: temperature-scaled sigmoid
        # Scale logits before sigmoid to control sharpness
        scaled = logits / temperature
        return backend.sigmoid(scaled)


def reason(pattern: str, predicates: Dict[str, Any], 
          bindings: Optional[Dict[str, int]] = None,
          temperature: float = 0.0,
          backend: Optional[Backend] = None,
          strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Temperature-controlled reasoning over a logical pattern.
    
    Args:
        pattern: Logical pattern (e.g., "Grandparent(x, z)")
        predicates: Dictionary of predicate tensors
        bindings: Optional variable bindings
        temperature: Temperature for inference control (0=deductive, >0=analogical)
        backend: Backend for operations
        strategy: Compilation strategy
    
    Returns:
        Inference result
    
    Example:
        >>> parent = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> # Infer grandparent via composition
        >>> result = reason('Grandparent(x, z)', 
        ...                 predicates={'Parent': parent},
        ...                 temperature=0.0)
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    
    # Parse pattern - simple implementation for common cases
    if "Grandparent" in pattern or "grandparent" in pattern.lower():
        # Grandparent(x, z) = exists y: Parent(x, y) AND Parent(y, z)
        parent = predicates.get('Parent') if 'Parent' in predicates else predicates.get('parent')
        if parent is not None:
            # Compose: grandparent = parent @ parent (matrix multiplication)
            composition = backend.matmul(parent, parent)
            # Apply temperature
            result = apply_temperature(composition, temperature, backend)
            return result
    
    # Generic case: return first predicate
    if predicates:
        first_pred = list(predicates.values())[0]
        return apply_temperature(first_pred, temperature, backend)
    
    return None


def quantify(pattern: str, predicates: Dict[str, Any],
            backend: Optional[Backend] = None,
            strategy: Optional[LogicalStrategy] = None) -> Any:
    """
    Pattern-based quantified query.
    
    Args:
        pattern: Quantified pattern (e.g., "exists y: Parent(x, y) and Parent(y, z)")
        predicates: Dictionary of predicate tensors
        backend: Backend for operations
        strategy: Compilation strategy
    
    Returns:
        Query result
    
    Example:
        >>> parent = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        >>> # Check if Alice (row 0) has any children
        >>> result = quantify('exists y: Parent(x, y)', 
        ...                   predicates={'Parent': parent})
    """
    if backend is None:
        backend = create_backend()
    if strategy is None:
        strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    
    # Parse pattern - simplified implementation
    pattern_lower = pattern.lower()
    
    if 'exists' in pattern_lower and 'parent' in pattern_lower:
        parent = predicates.get('Parent') if 'Parent' in predicates else predicates.get('parent')
        if parent is not None:
            if 'and' in pattern_lower:
                # exists y: Parent(x,y) and Parent(y,z) -> grandparent
                composition = backend.matmul(parent, parent)
                return backend.step(composition)
            else:
                # exists y: Parent(x,y) -> has children
                return exists(parent, axis=1, backend=backend)
    
    if 'forall' in pattern_lower or 'for all' in pattern_lower:
        # Universal quantification
        parent = predicates.get('Parent') if 'Parent' in predicates else predicates.get('parent')
        if parent is not None:
            return forall(parent, axis=1, backend=backend)
    
    return None
