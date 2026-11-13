"""The Haru web demo with an external LLM choosing the words (see bridge.py). Same flags as haru-server."""
import haru.server as haru_server
from bridge import BridgeBackend

haru_server.LocalBackend = BridgeBackend

if __name__ == "__main__":
    haru_server.main()
