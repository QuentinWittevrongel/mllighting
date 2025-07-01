import logging


class LoggerManager:
    """Main interface for logging.

    Standardize logging through the whole program.
    """

    ROOT = 'mllighting'

    _instance = None

    def __init__(self):
        """Initialize the class."""
        self._root_logger = logging.getLogger(LoggerManager.ROOT)
        self._root_logger.propagate = False
        self._root_logger.setLevel(logging.DEBUG)

        # stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(logging.DEBUG)
        # self._root_logger.addHandler(stream_handler)

    def __new__(cls):
        """Create an instance of the class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def root_logger(self) -> logging.Logger:
        """The root logger."""
        return self._root_logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get the logger parented under the main logger namespace.

        Args:
            name: The name of the logger to create.

        Returns:
            The logger.
        """
        logger_name = f'{LoggerManager.ROOT}.{name}'
        return logging.getLogger(logger_name)


# Initialize the logging.
log = LoggerManager()
