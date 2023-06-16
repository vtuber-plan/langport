import httpx

class AsyncHttpPool(object):
    def __init__(self) -> None:
        pass

    async def get(self, *args, **kwarg):
        client = httpx.AsyncClient()
        response = await client.get(*args, **kwarg)
        await client.aclose()
        return response
    
    async def post(self, *args, **kwarg):
        client = httpx.AsyncClient()
        response = await client.post(*args, **kwarg)
        await client.aclose()
        return response
    
    def stream(self, *args, **kwarg):
        with httpx.AsyncClient() as client:
            return client.stream(*args, **kwarg)

    async def aclose(self):
        pass