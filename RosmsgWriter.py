from typing import Any, Dict, Optional


class RosmsgWriter:
    def __init__(self, output) -> None:
        self.__writer = output
        self.__schema_ids: Dict[str, int] = {}
        self.__channel_ids: Dict[str, int] = {}
        