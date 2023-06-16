
import argparse
import logging
import os

from langport.routers.gateway.common import AppSettings

logger = logging.getLogger(__name__)

import sys
from streamlit.web import cli as stcli

if __name__ in ["__main__", "langport.service.gateway.cluster_monitor"]:
    parser = argparse.ArgumentParser(
        description="Langport Cluster Monitor."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    args = parser.parse_args()

    logger.debug(f"==== args ====\n{args}")

    app_settings = AppSettings(controller_address=args.controller_address)
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'cluster_monitor_app.py')

    sys.argv = ["streamlit", "run", filename]
    sys.exit(stcli.main())
