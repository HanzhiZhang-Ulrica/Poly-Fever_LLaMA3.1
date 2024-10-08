# Environment Setup

## 1. Codes

Download scripts:

```shell
git clone https://github.com/HanzhiZhang-Ulrica/Poly-Fever_LLaMA3.1.git
```

Create 2 directories: `data` and `model`

```shell
mkdir data
mkdir model
```

Now, the 1 level tree of project should be like:

```
.
├── data
├── model
├── requirements.txt
└── scripts
```

## 2. Datas

Download dataset of this project at `data` directory:

```shell
wget -O Poly-FEVER_all.tsv "https://www.dropbox.com/scl/fi/4ej8w4bc9giznwl8wlo4w/Poly-FEVER_all.tsv?rlkey=mwtx8x1ipxl948xq9x21plssr&st=25v3a0ie&dl=1"
```

Download dataset of this project at `script` directory:

```shell
wget -O 100_pickle_files.zip "https://www.dropbox.com/scl/fo/0fyy97jxhumykkmengzuk/ANyPNgZxa54EcP3TKqGQc0c?rlkey=9mndhlh24b0vuiwxm06oo2y1s&st=ev4m8qyq&dl=1"

unzip 100_pickle_files.zip -d 100_pickle_files
rm 100_pickle_files.zip 
```

Now, the 2 levels tree of project should be like:

```
.
├── data
│   └── Poly-FEVER_all.tsv
├── model
├── requirements.txt
└── scripts
    ├── 100_pickle_files
    ├── 1_hallu_observe.py
    ├── 2_improve_LDA.py
    ├── 2_improve_RAG.py
    ├── LDA
    ├── llm
    └── rag
```

## 3. Packages

```shell
pip install -r requirements.txt
```

## 4. Models

To download models and datas, should login hugging face:

```shell
pip install --upgrade huggingface_hub
huggingface-cli login
```

The login token is:

```
[your token]
```

The models will be automatically downloaded when the first processing of any `1_hallu_observe.py / 2_improve_LDA.py / 2_improve_RAG.py` script. When processing `2_improve_RAG.py`, also download other relevant models and data.
