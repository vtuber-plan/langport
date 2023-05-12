# langport
LangPort is a open-source large language model serving platform.
It is inspired by project [lmsys/fastchat](https://github.com/lm-sys/FastChat), we hope that the serving platform is lightweight and fast, but fastchat includes other features such as training and evaluation make it complicate.

The core features include:
- A distributed serving system for state-of-the-art models.
- OpenAI-Compatible RESTful APIs.
- FauxPilot-Compatible RESTful APIs.

## News
- [2023/05/10] langport project started.


## Install

### Method 1: With pip

```bash
pip3 install langport
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

``` bash
python -m langport.service.controller
python -m langport.service.embedding_worker
python -m langport.service.generation_worker
python -m langport.service.openai_api
python -m langport.service.fauxpilot_api
```

## License

langport is released under the MIT License.


## See also

- [langport-docs](https://langport.readthedocs.io/)
- [langport-source](https://github.com/vtuber-plan/langport)
