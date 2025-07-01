import logging

import hou

from mllighting import log


LOGGING_TO_SEVERITYTYPE_MAP = {
    logging.DEBUG: hou.severityType.Message,
    logging.INFO: hou.severityType.ImportantMessage,
    logging.WARNING: hou.severityType.Warning,
    logging.ERROR: hou.severityType.Error,
    logging.FATAL: hou.severityType.Fatal,
}


class HoudiniHandler(logging.Handler):
    """Handler that logs Python logging into the Houdini log system."""

    SOURCE_NAME = 'ML Lighting'

    def __init__(self):
        super().__init__()
        hou.logging.createSource(self.SOURCE_NAME)

    def emit(self, record: logging.LogRecord):
        severity = LOGGING_TO_SEVERITYTYPE_MAP.get(
            record.levelno, hou.severityType.ImportantMessage)
        entry = hou.logging.LogEntry(
            message=self.format(record),
            source=record.name,
            severity=severity,
            time=record.created)

        hou.logging.log(entry, source_name=self.SOURCE_NAME)


# Initialize the logger.
log_manager = log.LoggerManager()
houdini_handler = HoudiniHandler()
houdini_handler.setLevel(logging.DEBUG)
log_manager.root_logger.addHandler(houdini_handler)

logger = log.LoggerManager.get_logger(__name__)
logger.debug('Houdini log initialized')
