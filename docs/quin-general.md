# Quin-general

This is modified from [Quin+](https://github.com/algoprog/Quin).

The Quin-general backend model currently runs on A6000 workstation. 

## Quick start
The Quin-general process is currently managed as a systemd service, with the script located at `/etc/systemd/system/quin-general.service`.

To check if Quin-general is running, run the following command:
```bash
sudo systemctl status quin-general.service
```

If it is not running, you can start the service by running:
```bash
sudo systemctl restart quin-general.service
```

To run the script directly for testing, change the directory to then run `python quin_web.py --mode=serve`

## Set up from scratch

1. Download the encoder model from [here](https://drive.google.com/file/d/1qsDPreap_26mL3UFDEyVPoe9ygbniLx9/view?usp=sharing) or NUS Dropbox and put it under `models/weights/encoder`
2. Download the NLI model from [here](https://drive.google.com/file/d/15Txw44izeEHCzzXIpxwVXFvNz_-_kng-/view?usp=sharing) and put it under `models/weights/nli`
3. Download the passage ranking model from [here](https://drive.google.com/file/d/1D0cGpM2uDMWmnYmF35Tbuv0Z_yE4l0U8/view?usp=sharing) and put it under `models/weights/passage_ranker`
4. Run `pip install -r backend/requirements.txt`

### Setting up the GPU

In `backend/quin_web.py` you can find the following line:

`os.environ["CUDA_VISIBLE_DEVICES"] = "..."`

Depending on which GPUs are available on the machine, you should set this variable to at least one avaiable GPU id. One GPU is sufficient to run the system. For example, if GPU number 3 is available then you can set it as follows:

`os.environ["CUDA_VISIBLE_DEVICES"] = "3"`

A nice way to check the availability of the GPUs is using the `gpustat` command. You can install this tool using pip:

`pip install gpustat`
