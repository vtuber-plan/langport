# langport
LangPort is a open-source large language model serving platform.
It is inspired by project [lmsys/fastchat](https://github.com/lm-sys/FastChat), we hope that the serving platform is lightweight and fast, but fastchat includes other features such as training and evaluation make it complicate.

The core features include:
- A distributed serving system for state-of-the-art models.
- Streaming batch inference for higher throughput.
- OpenAI-Compatible RESTful APIs.
- FauxPilot-Compatible RESTful APIs.

## News
- [2023/05/10] langport project started.
- [2023/05/14] batch inference supported.


## Install

### Method 1: With pip

```bash
pip3 install git+https://github.com/vtuber-plan/langport.git 
```

### Method 2: From source

1. Clone this repository
```bash
git clone https://github.com/vtuber-plan/langport.git
cd langport
```

2. Install the Package
```bash
pip install --upgrade pip
pip install -e .
```

## Start the server

It is simple to start a local chat API service:
``` bash
python -m langport.service.controller
python -m langport.service.generation_worker --model-path <your model path>
python -m langport.service.openai_api
```

``` bash
python -m langport.service.controller
python -m langport.service.generation_worker --model-path <your model path>
python -m langport.service.embedding_worker --model-path <your model path>
python -m langport.service.openai_api
python -m langport.service.fauxpilot_api
```

## License

langport is released under the MIT License.


## See also

- [langport-docs](https://langport.readthedocs.io/)
- [langport-source](https://github.com/vtuber-plan/langport)
