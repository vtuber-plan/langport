
import argparse
import asyncio
import json
import logging
import httpx
import pandas as pd
import graphyte

from langport.protocol.worker_protocol import WorkerAddressRequest, WorkerAddressResponse
from langport.routers.gateway.common import AppSettings


logger = logging.getLogger(__name__)

async def list_workers(app_settings: AppSettings):
    data_dict = {
        "address": [],
        "features": [],
        "model": [],
        "queue_length": [],
    }
    async with httpx.AsyncClient() as client:
        payload = WorkerAddressRequest(
            condition="True", expression="{model_name}, {features}, {queue_length}"
        )
        ret = await client.post(
            app_settings.controller_address + "/get_worker_address",
            json=payload.dict(),
        )
        if ret.status_code != 200:
            return []
        response = WorkerAddressResponse.parse_obj(ret.json())
        address_list = response.address_list
        data = [json.loads(obj) for obj in response.values]

    for i in range(len(address_list)):
        data_dict["address"].append(address_list[i])
        data_dict["features"].append(str(data[i][1]))
        data_dict["model"].append(data[i][0])
        data_dict["queue_length"].append(data[i][2])
    
    return pd.DataFrame(data_dict)

async def main(app_settings):
    graphyte.init(args.graphite_address, prefix=args.prefix)

    while True:
        df = await list_workers(app_settings)
        for i, row in df.iterrows():
            address_name = row['address'].replace('.', '_')
            graphyte.send(f"langport.{address_name}.queue_length", row["queue_length"])

        await asyncio.sleep(3)


if __name__ in ["__main__", "langport.service.gateway.graphite_feeder"]:
    parser = argparse.ArgumentParser(
        description="Langport Cluster Graphite Data Monitor."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--graphite-address", type=str, default="graphite.example.com"
    )
    parser.add_argument(
        "--prefix", type=str, default="system.sync"
    )
    args = parser.parse_args()

    logger.debug(f"==== args ====\n{args}")

    app_settings = AppSettings(controller_address=args.controller_address)
    
    asyncio.run(main(app_settings))