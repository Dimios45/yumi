"""Read current GUI values from the local Viser viewer without changing them."""
import asyncio
import json
from pathlib import Path

import msgspec
import viser
import websockets
import zstandard


async def main():
    components = {}
    async with websockets.connect('ws://127.0.0.1:8082', subprotocols=[f'viser-v{viser.__version__}'], max_size=32*1024*1024) as ws:
        deadline = asyncio.get_running_loop().time()+3
        while asyncio.get_running_loop().time() < deadline:
            try:
                data = await asyncio.wait_for(ws.recv(), timeout=.5)
            except asyncio.TimeoutError:
                continue
            compressed_size = int.from_bytes(data[8:16], 'little')
            messages = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(data[16:16+compressed_size]))
            if isinstance(messages, dict) and 'messages' in messages:
                messages = messages['messages']
            if isinstance(messages, dict):
                messages = [messages]
            for m in messages:
                if 'uuid' not in m:
                    continue
                uid = m['uuid']
                if 'value' in m and 'props' in m:
                    components[uid] = {'label': m['props'].get('label'), 'value': m['value']}
                if m['type'] == 'GuiUpdateMessage' and uid in components and 'value' in m['updates']:
                    components[uid]['value'] = m['updates']['value']
    values = {c['label']: c['value'] for c in components.values() if c['label']}
    Path('artifacts/current-yam-viewer-state.json').write_text(json.dumps(values, indent=2)+'\n')
    print(json.dumps(values, indent=2))


asyncio.run(main())
