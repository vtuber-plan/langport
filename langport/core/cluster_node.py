import asyncio
from collections import defaultdict
import logging
import json
import time
from typing import Any, Dict, List

import httpx
from langport.core.base_node import BaseNode

from langport.protocol.worker_protocol import (
    GetNodeStateRequest,
    GetNodeStateResponse,
    NodeInfo,
    NodeInfoRequest,
    NodeInfoResponse,
    NodeListRequest,
    NodeListResponse,
    RegisterNodeRequest,
    RegisterNodeResponse,
    RemoveNodeRequest,
    RemoveNodeResponse,
    HeartbeatPing,
    HeartbeatPong,
)

from langport.constants import (
    HEART_BEAT_EXPIRATION,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils.cache_state import CacheState
from langport.utils.http_pool import AsyncHttpPool
from cachetools import LRUCache, TTLCache
from asyncache import cached

class ClusterNode(BaseNode):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        logger: logging.Logger,
    ):
        super(ClusterNode, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            logger=logger,
        )
        self.init_neighborhoods_addr: List[str] = init_neighborhoods_addr
        self.neighborhoods: Dict[str, NodeInfo] = {}
        self.headers = {"User-Agent": "LangPort nodes"}

        self.states: Dict[str, CacheState] = {}
        self.remote_states: Dict[str, Dict[str, CacheState]] = defaultdict(dict)

        self.client = AsyncHttpPool()
        
        # timers
        self.add_timer(
            "heartbeat",
            WORKER_HEART_BEAT_INTERVAL,
            self.send_heartbeat_broadcast,
            args=None,
            kwargs=None,
            workers=1,
        )

        self.add_timer(
            "expiration_check",
            HEART_BEAT_EXPIRATION // 2,
            self.remove_nodes_expiration,
            args=None,
            kwargs=None,
            workers=1,
        )

        # start and stop
        self.on_start("get_all_init_neighborhoods", self.get_all_init_neighborhoods)
        self.on_start("register_node_broadcast", self.register_self_node_broadcast)
        self.on_stop("remove_node_broadcast", self.remove_self_node_broadcast)
    
    def _add_node(self, node_id: str, node_addr: str, check_heart_beat: bool=True):
        self.neighborhoods[node_id] = NodeInfo(
            node_id=node_id,
            node_addr=node_addr,
            check_heart_beat=check_heart_beat,
            refresh_time=int(time.time())
        )
    
    def _update_node(self, node_id: str, node_addr: str, check_heart_beat: bool=True):
        self.neighborhoods[node_id] = NodeInfo(
            node_id=node_id,
            node_addr=node_addr,
            check_heart_beat=check_heart_beat,
            refresh_time=int(time.time())
        )
    
    def _remove_node(self, node_id: str):
        if node_id in self.neighborhoods:
            del self.neighborhoods[node_id]
    
    async def get_all_init_neighborhoods(self):
        for neighbor_addr in self.init_neighborhoods_addr:
            try:
                info = await self.get_node_info(neighbor_addr)
            except httpx.ReadTimeout:
                self.logger.error(f"Cannot connect to neighbor {neighbor_addr}.")
                continue
            self._add_node(info.node_id, info.node_addr, info.check_heart_beat)
        # add self
        self._add_node(self.node_id, self.node_addr, True)
    
    async def get_node_info(self, node_addr: str) -> NodeInfo:
        response = await self.client.post(
            node_addr + "/node_info",
            headers=self.headers,
            json=NodeInfoRequest(node_id=self.node_id).dict(),
            timeout=WORKER_API_TIMEOUT,
        )
        remote_node_info = NodeInfoResponse.parse_obj(response.json())
        return remote_node_info.node_info

    async def register_node(self, target_node_addr: str, register_node_id: str, register_node_addr: str) -> bool:
        self.logger.info(f"Register {register_node_addr} to node {target_node_addr}")

        if target_node_addr == self.node_addr:
            self._add_node(register_node_id, register_node_addr)
            return True
        else:
            data = RegisterNodeRequest(
                node_id=register_node_id,
                node_addr=register_node_addr,
                check_heart_beat=True,
            )
        
            response = await self.client.post(
                target_node_addr + "/register_node",
                headers=self.headers,
                json=data.dict(),
                timeout=WORKER_API_TIMEOUT,
            )
            remote = RegisterNodeResponse.parse_obj(response.json())
            self._add_node(remote.node_id, remote.node_addr)
            
            # fetch remote node info
            remote_nodes = await self.fetch_all_nodes(remote.node_addr)
            for node in remote_nodes.nodes:
                self._update_node(node_id=node.node_id, node_addr=node.node_addr)
            return True

    async def register_node_broadcast(self, register_node_id: str, register_node_addr: str):
        self.logger.info(f"Register node broadcast. node_id: {register_node_id}, node_addr: {register_node_addr}")

        neighborhoods = [(k, v) for k, v in self.neighborhoods.items()]
        for node_id, node_info in neighborhoods:
            if node_info.node_addr == register_node_addr:
                continue
            await self.register_node(node_info.node_addr, register_node_id, register_node_addr)
    
    async def register_self_node_broadcast(self):
        await self.register_node_broadcast(self.node_id, self.node_addr)

    async def remove_node(self, target_node_addr: str, removed_node_id: str) -> bool:
        self.logger.info(f"Remove node {removed_node_id} from {target_node_addr}")

        if target_node_addr == self.node_addr:
            return True

        data = RemoveNodeRequest(
            node_id=removed_node_id,
        )

        response = await self.client.post(
            target_node_addr + "/remove_node",
            headers=self.headers,
            json=data.dict(),
            timeout=WORKER_API_TIMEOUT,
        )
        ret = RemoveNodeResponse.parse_obj(response.json())

        return True

    async def remove_node_broadcast(self, removed_node_id: str):
        self.logger.info(f"Remove node broadcast. node_addr: {removed_node_id}")
        neighborhoods = [(k, v) for k, v in self.neighborhoods.items()]
        for node_id, node_info in neighborhoods:
            if node_id == removed_node_id:
                continue
            await self.remove_node(node_info.node_addr, removed_node_id)

    async def remove_self_node_broadcast(self):
        await self.remove_node_broadcast(self.node_id)

    async def send_heartbeat(self, node_addr: str):
        data = HeartbeatPing(
            node_id=self.node_id,
        )
        response = await self.client.post(
            node_addr + "/heartbeat",
            headers=self.headers,
            json=data.dict(),
            timeout=WORKER_API_TIMEOUT,
        )
        ret = HeartbeatPong.parse_obj(response.json())
  
        return ret

    async def send_heartbeat_broadcast(self):
        self.logger.info(
            f"Send heartbeat. Worker: {self.node_id}; Address: {self.node_addr}."
        )

        self.logger.info(
            f"Neighborhoods: {[i for i, v in self.neighborhoods.items()]}."
        )

        neighborhoods = {k: v for k, v in self.neighborhoods.items()}
        for node_id, node_info in neighborhoods.items():
            if node_id == self.node_id:
                continue
            await self.send_heartbeat(node_info.node_addr)
        
    async def fetch_all_nodes(self, node_addr: str):
        data = NodeListRequest(
            node_id=self.node_id,
        )

        response = await self.client.post(
            node_addr + "/node_list",
            headers=self.headers,
            json=data.dict(),
            timeout=WORKER_API_TIMEOUT,
        )
        ret = NodeListResponse.parse_obj(response.json())
  
        return ret

    def remove_nodes_expiration(self):
        neighborhoods = [(k, v) for k, v in self.neighborhoods.items()]
        for node_id, node_info in neighborhoods:
            if node_id == self.node_id:
                continue
            if time.time() - node_info.refresh_time > HEART_BEAT_EXPIRATION:
                self.logger.info(f"Node {node_id} is expired.")
                self._remove_node(node_id)

    @cached(TTLCache(maxsize=1024, ttl=16))
    async def request_node_state(self, node_addr: str, name: str) -> GetNodeStateResponse:
        data = GetNodeStateRequest(
            state_name=name,
        )

        response = await self.client.post(
            node_addr + "/get_node_state",
            headers=self.headers,
            json=data.dict(),
            timeout=WORKER_API_TIMEOUT,
        )
        ret = GetNodeStateResponse.parse_obj(response.json())

        return ret
    
    async def _get_node_state_local(self, node_id: str, name: str) -> Any:
        entry = self.states.get(name, None)
        if entry is not None:
            return entry.get()
        else:
            return None
    
    @cached(TTLCache(maxsize=1024, ttl=1))
    async def _get_node_state_remote(self, node_id: str, name: str) -> Any:
        target_state = self.remote_states[node_id]
        if name in target_state:
            if target_state[name].is_valid():
                return target_state[name].get()

        node_info = self.neighborhoods[node_id]
        response = await self.request_node_state(node_info.node_addr, name)
        value = json.loads(response.state_value)
        ttl = response.state_ttl
        if value is not None:
            # print(f"refresh state {name}, node id {node_id}, value {value}")
            self.remote_states[node_id][name] = CacheState(value, ttl)
            return self.remote_states[node_id][name].get()
        else:
            return None
    
    async def get_node_state(self, node_id: str, name: str) -> Any:
        if node_id == self.node_id:
            return await self._get_node_state_local(node_id, name)
        else:
            state_lock = asyncio.Lock()
            async with state_lock:
                ret = await self._get_node_state_remote(node_id, name)
            return ret
    
    async def set_local_state(self, name: str, value: Any, ttl: int=60):
        self.states[name] = CacheState(value, ttl)

    async def api_register_node(self, request: RegisterNodeRequest) -> RegisterNodeResponse:
        if request.node_id not in self.neighborhoods:
            self.logger.info(f"Register {request.node_id} on {self.node_id}")
            self._add_node(request.node_id, request.node_addr)

            # broacast again
            await self.register_node_broadcast(request.node_id, request.node_addr)
        
        return RegisterNodeResponse(
            node_id=self.node_id,
            node_addr=self.node_addr,
            check_heart_beat=True
        )

    async def api_remove_node(self, request: RemoveNodeRequest) -> RemoveNodeResponse:
        if request.node_id in self.neighborhoods:
            self.logger.info(f"Remove {request.node_id} from {self.node_id}")
            self._remove_node(request.node_id)

            # broacast again
            await self.remove_node_broadcast(request.node_id)
        return RemoveNodeResponse(node_id=self.node_id)
    
    async def api_receive_heartbeat(self, request: HeartbeatPing) -> HeartbeatPong:
        if request.node_id in self.neighborhoods:
            self.neighborhoods[request.node_id].refresh_time = int(time.time())
        else:
            self.logger.info(f"Invalid ping packet from {request.node_id}.")
        return HeartbeatPong(exist=True)
    
    async def api_return_node_list(self, request: NodeListRequest) -> NodeListResponse:
        node_list = [node_info for node_id, node_info in self.neighborhoods.items()]
        return NodeListResponse(nodes=node_list)
    
    async def api_return_node_info(self, request: NodeInfoRequest) -> NodeInfoResponse:
        return NodeInfoResponse(
            node_info=NodeInfo(
                node_id=self.node_id,
                node_addr=self.node_addr,
                check_heart_beat=True,
            )
        )
    
    async def api_return_node_state(self, request: GetNodeStateRequest) -> GetNodeStateResponse:
        if request.state_name in self.states:
            entry = self.states[request.state_name]
            return GetNodeStateResponse(
                state_value=json.dumps(entry.value),
                state_ttl=entry.ttl
            )
        else:
            return GetNodeStateResponse(
                state_value=json.dumps(None)
            )