from fastapi import FastAPI, BackgroundTasks
from langport.protocol.worker_protocol import GetNodeStateRequest, HeartbeatPing, NodeInfoRequest, NodeListRequest, RegisterNodeRequest, RemoveNodeRequest, WorkerAddressRequest

app = FastAPI()

def create_background_tasks(worker):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: worker.release_model_semaphore())
    return background_tasks

@app.on_event("startup")
async def startup_event():
    await app.node.start()


@app.on_event("shutdown")
async def shutdown_event():
    await app.node.stop()


@app.post("/register_node")
async def register_node(request: RegisterNodeRequest):
    response = await app.node.api_register_node(request)
    return response.dict()


@app.post("/remove_node")
async def remove_node(request: RemoveNodeRequest):
    response = await app.node.api_remove_node(request)
    return response.dict()


@app.post("/heartbeat")
async def receive_heartbeat(request: HeartbeatPing):
    response = await app.node.api_receive_heartbeat(request)
    return response.dict()


@app.post("/node_list")
async def return_node_list(request: NodeListRequest):
    response = await app.node.api_return_node_list(request)
    return response.dict()


@app.post("/node_info")
async def return_node_info(request: NodeInfoRequest):
    response = await app.node.api_return_node_info(request)
    return response.dict()

@app.post("/get_node_state")
async def api_return_node_state(request: GetNodeStateRequest):
    response = await app.node.api_return_node_state(request)
    return response.dict()

@app.post("/get_worker_address")
async def api_get_worker_address(request: WorkerAddressRequest):
    response = await app.node.api_get_worker_address(request)
    return response.dict()

