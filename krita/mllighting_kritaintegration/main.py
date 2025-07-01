import krita

from PyQt5 import QtWidgets

from mllighting import log

from mllighting_kritaintegration import commands, server


logger = log.LoggerManager.get_logger(__name__)


class MLLightingDocker(krita.DockWidget):

    def __init__(self):
        super().__init__()

        self._server_manager = server.KritaServerManager()

        self.setWindowTitle('Ml Lighting')

        main_widget = QtWidgets.QWidget(self)
        self.setWidget(main_widget)

        main_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_layout)

        self.server_lineedit = QtWidgets.QLineEdit()
        main_layout.addWidget(self.server_lineedit)
        self.server_lineedit.setText('127.0.0.1')

        self.port_lineedit = QtWidgets.QLineEdit()
        main_layout.addWidget(self.port_lineedit)
        self.port_lineedit.setText('8002')

        start_button = QtWidgets.QPushButton('Start server')
        main_layout.addWidget(start_button)

        stop_button = QtWidgets.QPushButton('Stop server')
        main_layout.addWidget(stop_button)

        self.lightingappserver_lineedit = QtWidgets.QLineEdit()
        main_layout.addWidget(self.lightingappserver_lineedit)
        self.lightingappserver_lineedit.setText('127.0.0.1')

        self.lightingappport_lineedit = QtWidgets.QLineEdit()
        main_layout.addWidget(self.lightingappport_lineedit)
        self.lightingappport_lineedit.setText('8001')

        send_button = QtWidgets.QPushButton('Send drawing')
        main_layout.addWidget(send_button)

        start_button.clicked.connect(self._start_button_clicked)
        stop_button.clicked.connect(self._stop_button_clicked)

        send_button.clicked.connect(self._send_button_clicked)

        # Register the commands.
        self._server_manager.register_command(
            'send_albedo', commands.albedo_received)

    def canvasChanged(self, canvas: krita.Canvas):
        pass

    def _start_button_clicked(self, checked: bool):
        """Method executed when the start button is clicked.

        Args:
            checked: If the button is checked.
        """
        address = self.server_lineedit.text()
        port = int(self.port_lineedit.text())

        self._server_manager.start_server(address, port)

    def _stop_button_clicked(self, checked: bool):
        """Method executed when the stop button is clicked.

        Args:
            checked: If the button is checked.
        """
        self._server_manager.stop_server()

    def _send_button_clicked(self, checked: bool):
        """Method executed when the send button is clicked.

        Args:
            checked: If the button is checked.
        """
        document = krita.Krita.instance().activeDocument()
        address = self.lightingappserver_lineedit.text()
        port = int(self.lightingappport_lineedit.text())
        commands.send_beauty(document, address, port, self._server_manager)


# Initialize the Krita docker.
krita_instance = krita.Krita.instance()
dockerwidget_factory = krita.DockWidgetFactory(
    'mllighting',
    krita.DockWidgetFactoryBase.DockTornOff,
    MLLightingDocker)
krita.Krita.instance().addDockWidgetFactory(dockerwidget_factory)
