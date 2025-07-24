from channels.generic.websocket import AsyncWebsocketConsumer
import json

class DatasetProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.dataset_id = self.scope['url_route']['kwargs']['dataset_id']
        self.group_name = f'dataset_progress_{self.dataset_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def progress_update(self, event):
        await self.send(text_data=json.dumps({
            'progress': event['progress'],
            'message': event['message'],
        })) 