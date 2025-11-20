"""Dependency injection container for managing application dependencies.

This module provides a lightweight dependency injection container that
manages the lifecycle and dependencies of application components.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar
import threading
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """Dependency injection container.

    This container manages service instances and their dependencies,
    supporting both singleton and transient lifetimes.

    Examples:
        >>> container = Container()
        >>>
        >>> # Register a singleton service
        >>> container.register_singleton(IDataLoader, DataLoader)
        >>>
        >>> # Register a transient service
        >>> container.register_transient(IDataValidator, DataValidator)
        >>>
        >>> # Resolve a service
        >>> data_loader = container.resolve(IDataLoader)
    """

    def __init__(self):
        """Initialize the container."""
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Callable] = {}
        self._factories: Dict[Type, Callable] = {}
        self._lock = threading.Lock()

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        instance: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a singleton service.

        Only one instance will be created and shared across the application.

        Args:
            interface: The interface or base class
            implementation: The concrete implementation class (optional if instance or factory provided)
            instance: An already created instance (optional)
            factory: A factory function that creates the instance (optional)

        Examples:
            >>> # Register with implementation class
            >>> container.register_singleton(IDataLoader, DataLoader)
            >>>
            >>> # Register with instance
            >>> loader = DataLoader(config)
            >>> container.register_singleton(IDataLoader, instance=loader)
            >>>
            >>> # Register with factory
            >>> container.register_singleton(IDataLoader, factory=lambda: DataLoader(config))
        """
        with self._lock:
            if instance is not None:
                self._singletons[interface] = instance
                logger.debug(f"Registered singleton instance for {interface.__name__}")
            elif factory is not None:
                self._factories[interface] = factory
                logger.debug(f"Registered singleton factory for {interface.__name__}")
            elif implementation is not None:
                self._singletons[interface] = implementation
                logger.debug(f"Registered singleton class for {interface.__name__}")
            else:
                raise ValueError("Must provide implementation, instance, or factory")

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service.

        A new instance will be created each time it's resolved.

        Args:
            interface: The interface or base class
            implementation: The concrete implementation class

        Examples:
            >>> container.register_transient(IDataValidator, DataValidator)
        """
        with self._lock:
            self._transients[interface] = implementation
            logger.debug(f"Registered transient for {interface.__name__}")

    def resolve(self, interface: Type[T], *args, **kwargs) -> T:
        """Resolve a service instance.

        Args:
            interface: The interface or class to resolve
            *args: Arguments to pass to the constructor (for transients)
            **kwargs: Keyword arguments to pass to the constructor (for transients)

        Returns:
            Instance of the requested service

        Raises:
            ValueError: If the service is not registered

        Examples:
            >>> data_loader = container.resolve(IDataLoader)
        """
        # Check singletons first
        if interface in self._singletons:
            instance = self._singletons[interface]

            # If it's a class, instantiate it
            if isinstance(instance, type):
                with self._lock:
                    if isinstance(self._singletons[interface], type):
                        self._singletons[interface] = instance(*args, **kwargs)
                    return self._singletons[interface]

            return instance

        # Check factories
        if interface in self._factories:
            with self._lock:
                if interface not in self._singletons:
                    factory = self._factories[interface]
                    self._singletons[interface] = factory()
                return self._singletons[interface]

        # Check transients
        if interface in self._transients:
            implementation = self._transients[interface]
            return implementation(*args, **kwargs)

        raise ValueError(f"Service not registered: {interface.__name__}")

    def try_resolve(self, interface: Type[T], *args, **kwargs) -> Optional[T]:
        """Try to resolve a service, returning None if not registered.

        Args:
            interface: The interface or class to resolve
            *args: Arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            Instance of the requested service or None if not registered
        """
        try:
            return self.resolve(interface, *args, **kwargs)
        except ValueError:
            return None

    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered.

        Args:
            interface: The interface to check

        Returns:
            True if the service is registered, False otherwise
        """
        return (
            interface in self._singletons or
            interface in self._transients or
            interface in self._factories
        )

    def clear(self) -> None:
        """Clear all registrations.

        Warning: This will remove all registered services.
        """
        with self._lock:
            self._singletons.clear()
            self._transients.clear()
            self._factories.clear()
            logger.warning("Container cleared")

    def __repr__(self) -> str:
        """String representation of the container."""
        num_singletons = len(self._singletons) + len(self._factories)
        num_transients = len(self._transients)
        return f"<Container singletons={num_singletons} transients={num_transients}>"


# Global container instance
_global_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the global container instance.

    Returns:
        Global Container instance

    Examples:
        >>> from src.core import get_container
        >>> container = get_container()
        >>> data_loader = container.resolve(IDataLoader)
    """
    global _global_container

    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = Container()

    return _global_container


def inject(interface: Type[T]) -> Callable:
    """Decorator for automatic dependency injection.

    This decorator automatically resolves and injects dependencies from the container.

    Args:
        interface: The interface to inject

    Returns:
        Decorator function

    Examples:
        >>> @inject(IDataLoader)
        ... def process_data(data_loader: IDataLoader):
        ...     data = data_loader.load_all_datasets()
        ...     # Process data...
        >>>
        >>> # Function will be called with data_loader injected from container
        >>> process_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            dependency = container.resolve(interface)
            return func(dependency, *args, **kwargs)
        return wrapper
    return decorator


def reset_container() -> None:
    """Reset the global container.

    Warning: This should only be used in tests.
    """
    global _global_container
    with _container_lock:
        if _global_container is not None:
            _global_container.clear()
        _global_container = None
