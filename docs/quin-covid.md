# Quin-COVID

The Quin-COVID backend model currently runs on the A6000 workstation. It requires at least 8GB RAM and one GPU with 4GB+ VRAM.

## Quick start

The Quin-COVID process is currently managed as a systemd service, with the script located at `/etc/systemd/system/quin-backend-covid.service`.

To check if Quin-COVID is running, run the following command:
```bash
sudo systemctl status quin-backend-covid
```

If it is not running, you can start the service by running:
```bash
sudo systemctl restart quin-backend-covid
```

- Below was the old setup on the deprecated A100 machine.
```
The Quin-COVID process is currently managed as a system service, with the script located at `/etc/init.d/quin-covid`.

To check if Quin-COVID is running, run the following command:

$ /etc/init.d/quin-covid status


If it is not running, you can start the service by running:

$ /etc/init.d/quin-covid start


To run the script directly for testing, change the directory to then run `quin_covid.py --mode serve`
```

## Set up from scratch

1. Download the encoder model from [here](https://drive.google.com/file/d/1qsDPreap_26mL3UFDEyVPoe9ygbniLx9/view?usp=sharing) or NUS Dropbox,  and put it under `models/weights/encoder`
2. Download the NLI model from [here](https://drive.google.com/file/d/15Txw44izeEHCzzXIpxwVXFvNz_-_kng-/view?usp=sharing) and put it under `models/weights/nli`
3. Download the passage ranking model from [here](https://drive.google.com/file/d/1D0cGpM2uDMWmnYmF35Tbuv0Z_yE4l0U8/view?usp=sharing) and put it under `models/weights/passage_ranker`
4. Run `pip install -r backend/requirements.txt`

### Setting up the GPU

In `backend/quin_covid.py` you can find the following line:

`os.environ["CUDA_VISIBLE_DEVICES"] = "..."`

Depending on which GPUs are available on the machine, you should set this variable to at least one avaiable GPU id. One GPU is sufficient to run the system. For example, if GPU number 3 is available then you can set it as follows:

`os.environ["CUDA_VISIBLE_DEVICES"] = "3"`

A nice way to check the availability of the GPUs is using the `gpustat` command. You can install this tool using pip:

`pip install gpustat`

### Dataset preparation

There are existing scripts `scripts/update-quin-covid.news.sh` and `scripts/update-quin-covid-research.sh` which facilitate dataset updates. The exact steps are explained below:

1) To crawl new articles, run `backend/crawler.py` to update `backend/data/covid-news.jsonl`.

2) To create a new dataset of research papers based on the CORD corpus:
   - Download the latest CORD dataset [here](https://www.semanticscholar.org/cord19/download).
   - Place it under `backend/data/cord/`.
   - Run `backend/data/cord/build_dataset.py` to generate a new `backend/data/cord/dataset.jsonl` file.

3) Once the data files are in place, run `preprocessing/preprocess_encode.py --mode index` for scientific articles, or `preprocessing/preprocess_encode.py --mode index-news` for news, to encode and save the indexed data in the database.
