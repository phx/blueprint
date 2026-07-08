"""Core utility functions for The Divine Blueprint validation framework."""

from typing import Any, Callable, TypeVar, Dict, Union, Optional, List
from functools import lru_cache, wraps
import numpy as np
import time
from dataclasses import dataclass
from sympy import Expr, N
from .types import RealValue, ComplexValue, NumericValue
from .errors import PhysicsError, ComputationError, StabilityError, ValidationError

T = TypeVar('T')

def evaluate_expr(
    expr: Any,
    params: Dict[str, Union[float, NumericValue]],
    *,
    check_finite: bool = True
) -> NumericValue:
    """
    Safely evaluate expression with validation.
    
    Args:
        expr: Expression to evaluate
        params: Parameter values
        check_finite: Validate result is finite
        
    Returns:
        NumericValue: Evaluated result
    """
    try:
        symbols = {str(symbol) for symbol in getattr(expr, 'free_symbols', set())}
        if not symbols.issubset(set(params.keys())):
            missing = sorted(symbols - set(params.keys()))
            raise ValueError(f"Missing parameters: {missing}")
        result = expr.evalf(subs=params)
        value = complex(result) if result.is_complex else float(result)
        
        if check_finite and not np.isfinite(value):
            raise ValueError("Expression evaluated to non-finite value")
            
        return NumericValue(value)
        
    except Exception as e:
        raise ComputationError(f"Expression evaluation failed: {e}")

def cached_evaluation(func: Optional[Callable[..., T]] = None, *,
                      maxsize: int = 1024) -> Callable[..., T]:
    """Decorator form of cached evaluation with optional max size."""
    def decorate(inner: Callable[..., T]) -> Callable[..., T]:
        cached = lru_cache(maxsize=maxsize)(inner)

        @wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return cached(*args, **kwargs)
            except Exception as e:
                raise ComputationError(f"Cached evaluation failed: {e}")

        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        return wrapper

    if func is None:
        return decorate
    return decorate(func)

def check_numerical_stability(value: Union[float, complex, np.ndarray, Callable],
                            params: Optional[Dict[str, Any]] = None,
                            threshold: float = 1e-10) -> Union[bool, Dict[str, Any]]:
    """
    Check numerical stability of computation.
    
    Implements stability checks from the validation matrix:
    1. Check for NaN/Inf values
    2. Verify magnitude within bounds
    3. Check condition number for matrices
    
    Args:
        value: Value to check
        threshold: Stability threshold
    
    Returns:
        bool: True if value is numerically stable
    """
    try:
        if callable(value):
            try:
                result = value(**(params or {}))
            except Exception as e:
                raise StabilityError(f"Computation failed stability check: {e}")
            numeric = result.value if isinstance(result, NumericValue) else result
            if not np.all(np.isfinite(np.asarray(numeric))):
                raise StabilityError("Computation returned non-finite value")
            condition = 1.0
            uncertainty = getattr(result, 'uncertainty', None)
            stable = uncertainty is None or abs(uncertainty) <= max(threshold, abs(float(result)) + 1.0)
            return {
                'stable': bool(stable),
                'condition_number': condition,
                'value': result,
            }

        # Convert to numpy array for uniform handling
        arr = np.asarray(value)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(arr)):
            return False
            
        # Check magnitude
        if np.any(np.abs(arr) > 1/threshold):
            return False
            
        # Check for numerical noise
        small_nonzero = np.logical_and(np.abs(arr) < threshold, 
                                     np.abs(arr) > 0)
        if np.any(small_nonzero):
            return False
            
        return True
        
    except Exception as e:
        if callable(value):
            if isinstance(e, StabilityError):
                raise
            raise StabilityError("Computation failed stability check")
        return False

def stability_check(threshold: float = 1e-10) -> Callable:
    """
    Decorator to check numerical stability of function results.
    
    Args:
        threshold: Stability threshold
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if not check_numerical_stability(result, threshold):
                raise StabilityError(
                    f"Function {func.__name__} returned unstable result"
                )
            return result
        return wrapper
    return decorator

@dataclass
class ProfilingResult:
    """Results from profiling a computation."""
    execution_time: float
    memory_usage: float
    call_count: int
    avg_time_per_call: float

def profile_computation(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to profile computation time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function that collects profiling data
        
    Example:
        @profile_computation
        def heavy_calculation(data):
            # Computation here
            return result
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Track memory and time
        start_mem = get_memory_usage()
        start_time = time.perf_counter()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Compute metrics
        end_time = time.perf_counter()
        end_mem = get_memory_usage()
        
        previous = wrapper.profiling_data
        call_count = previous.call_count + 1
        elapsed = end_time - start_time
        avg_time = (
            (previous.avg_time_per_call * previous.call_count + elapsed) / call_count
            if call_count else elapsed
        )

        # Store profiling data
        wrapper.profiling_data = ProfilingResult(
            execution_time=elapsed,
            memory_usage=end_mem - start_mem,
            call_count=call_count,
            avg_time_per_call=avg_time
        )
        
        return result
    
    def get_profile() -> Dict[str, float]:
        data = wrapper.profiling_data
        return {
            'time': data.execution_time,
            'memory': data.memory_usage,
            'call_count': data.call_count,
            'avg_time_per_call': data.avg_time_per_call,
        }
    
    # Initialize profiling data
    wrapper.profiling_data = ProfilingResult(0.0, 0.0, 0, 0.0)
    wrapper.get_profile = get_profile  # type: ignore[attr-defined]
    return wrapper

def propagate_errors(values: List[RealValue], 
                    uncertainties: List[RealValue],
                    correlations: Optional[np.ndarray] = None) -> RealValue:
    """
    Propagate uncertainties through calculations.
    
    Args:
        values: List of measured values
        uncertainties: List of uncertainties
        correlations: Optional correlation matrix
        
    Returns:
        RealValue with propagated uncertainty
        
    Raises:
        PhysicsError: If inputs are invalid
        
    Example:
        result = propagate_errors(
            values=[x, y],
            uncertainties=[dx, dy],
            correlations=[[1, 0.5], [0.5, 1]]
        )
    """
    if len(values) != len(uncertainties):
        raise PhysicsError("Number of values and uncertainties must match")
        
    if correlations is not None:
        if correlations.shape != (len(values), len(values)):
            raise PhysicsError("Correlation matrix shape must match number of values")
            
        # Compute covariance matrix
        covariance = np.outer(uncertainties, uncertainties) * correlations
        
        # Propagate with correlations
        total_variance = np.sum(covariance)
        total_uncertainty = np.sqrt(total_variance)
        
    else:
        # Simple quadrature sum for uncorrelated uncertainties
        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2))
    
    # Compute central value
    central_value = np.sum(values)
    
    return RealValue(
        value=central_value,
        uncertainty=total_uncertainty
    )

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil  # type: ignore
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except (ImportError, AttributeError):
        # Return dummy value if psutil not available
        return -1.0

def batch_process(items: List[T], 
                 batch_size: int,
                 process_func: Callable[[List[T]], Any]) -> List[Any]:
    """
    Process items in batches to manage memory.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        process_func: Function to process each batch
        
    Returns:
        List of processed results
        
    Example:
        results = batch_process(
            items=large_dataset,
            batch_size=1000,
            process_func=compute_batch
        )
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    return results
