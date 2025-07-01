import logging

from PyQt5 import QtCore

from mllighting import log


LOGGING_TO_QMESSAGE_MAP = {
    logging.DEBUG: QtCore.qDebug,
    logging.INFO: QtCore.qInfo,
    logging.WARNING: QtCore.qWarning,
    logging.ERROR: QtCore.qFatal,
    logging.FATAL: QtCore.qCritical
}


class KritaHandler(logging.Handler):
    """Handler that logs Python logging into the Krita log system."""

    def emit(self, record: logging.LogRecord):
        qfunction = LOGGING_TO_QMESSAGE_MAP.get(record.levelno, QtCore.qInfo)
        msg = self.format(record)
        qfunction(msg)


# Initialize the logger.
log_manager = log.LoggerManager()
krita_handler = KritaHandler()
krita_handler.setLevel(logging.DEBUG)
log_manager.root_logger.addHandler(krita_handler)

logger = log.LoggerManager.get_logger(__name__)
logger.debug('Krita log initialized')
