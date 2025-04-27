from gnuradio.grc.core.base import Element

from gnuradio_mcp.models import ErrorModel
from gnuradio_mcp.utils import format_error_message


class ElementMiddleware:
    def __init__(self, element: Element):
        self._element = element

    def _rewrite(self):
        self._element.rewrite()

    def validate(self):
        self._rewrite()
        self._element.validate()

    def get_all_errors(self) -> list[ErrorModel]:
        self.validate()
        return [
            format_error_message(elem, msg)
            for elem, msg in self._element.iter_error_messages()
        ]
