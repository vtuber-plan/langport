import json
import pandas as pd
import streamlit as st
import asyncio
import httpx
from langport.protocol.worker_protocol import WorkerAddressRequest, WorkerAddressResponse

from langport.routers.gateway.common import AppSettings, _list_models

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
    
    df = pd.DataFrame(data_dict)
    return df

async def main():
    app_settings = AppSettings(controller_address="http://localhost:21001")
    st.title('LangPort Cluster Monitor')

    df = await list_workers(app_settings)
    st.table(df)
if __name__ == "__main__":
    asyncio.run(main())