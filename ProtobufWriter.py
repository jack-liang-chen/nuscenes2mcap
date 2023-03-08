from typing import Any, Dict, Optional
from mcap_protobuf.schema import register_schema


class ProtobufWriter:
    def __init__(self, output) -> None:
        self.__writer = output
        self.__schema_ids: Dict[str, int] = {}
        self.__channel_ids: Dict[str, int] = {}
    
    def write_message(
        self,
        topic: str,
        message: Any,
        log_time: Optional[int] = None,
        publish_time: Optional[int] = None,
        sequence: int = 0,
    ):
        """
        Writes a message to the MCAP stream, automatically registering schemas and channels as needs.
        """        
    
    