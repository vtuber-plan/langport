import httpx

class AsyncHttpPool(object):
    def __init__(self) -> None:
        self.client = httpx.AsyncClient()

    async def get(self, *args, **kwarg):
        response = await self.client.get(*args, **kwarg)
        return response
    
    async def post(self, *args, **kwarg):
        response = await self.client.post(*args, **kwarg)
        return response
    
    def stream(self, *args, **kwarg):
        return self.client.stream(*args, **kwarg)

    async def aclose(self):
        await self.client.aclose()
        self.client = None