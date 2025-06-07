## Quick Start

If you encounter any installation issues, please first consult the <a href="#faq">FAQ</a>. </br>
If the parsing results are not as expected, refer to the <a href="#known-issues">Known Issues</a>. </br>
There are three different ways to experience MinerU:

- [Online Demo (No Installation Required)](#online-demo)
- [Quick CPU Demo (Windows, Linux, Mac)](#quick-cpu-demo)
- Accelerate inference by using CUDA/CANN/MPS
    - [Linux/Windows + CUDA](#Using-GPU)
    - [Linux + CANN](#using-npu)
    - [MacOS + MPS](#using-mps)

> [!WARNING]
> **Pre-installation Notice—Hardware and Software Environment Support**
>
> To ensure the stability and reliability of the project, we only optimize and test for specific hardware and software
> environments during development. This ensures that users deploying and running the project on recommended system
> configurations will get the best performance with the fewest compatibility issues.
>
> By focusing resources on the mainline environment, our team can more efficiently resolve potential bugs and develop
> new features.
>
> In non-mainline environments, due to the diversity of hardware and software configurations, as well as third-party
> dependency compatibility issues, we cannot guarantee 100% project availability. Therefore, for users who wish to use
> this project in non-recommended environments, we suggest carefully reading the documentation and FAQ first. Most issues
> already have corresponding solutions in the FAQ. We also encourage community feedback to help us gradually expand
> support.

<table>
    <tr>
        <td colspan="3" rowspan="2">Operating System</td>
    </tr>
    <tr>
        <td>Linux after 2019</td>
        <td>Windows 10 / 11</td>
        <td>macOS 11+</td>
    </tr>
    <tr>
        <td colspan="3">CPU</td>
        <td>x86_64 / arm64</td>
        <td>x86_64(unsupported ARM Windows)</td>
        <td>x86_64 / arm64</td>
    </tr>
    <tr>
        <td colspan="3">Memory Requirements</td>
        <td colspan="3">16GB or more, recommended 32GB+</td>
    </tr>
    <tr>
        <td colspan="3">Storage Requirements</td>
        <td colspan="3">20GB or more, with a preference for SSD</td>
    </tr>
    <tr>
        <td colspan="3">Python Version</td>
        <td colspan="3">3.10~3.13</td>
    </tr>
    <tr>
        <td colspan="3">Nvidia Driver Version</td>
        <td>latest (Proprietary Driver)</td>
        <td>latest</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CUDA Environment</td>
        <td colspan="2"><a href="https://pytorch.org/get-started/locally/">Refer to the PyTorch official website</a></td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CANN Environment(NPU support)</td>
        <td>8.0+(Ascend 910b)</td>
        <td>None</td>
        <td>None</td>
    </tr>
    <tr>
        <td rowspan="2">GPU/MPS Hardware Support List</td>
        <td colspan="2">GPU VRAM 6GB or more</td>
        <td colspan="2">All GPUs with Tensor Cores produced from Volta(2017) onwards.<br>
        More than 6GB VRAM </td>
        <td rowspan="2">Apple silicon</td>
    </tr>
</table>

### Online Demo

Synced with dev branch updates:

[![OpenDataLab](https://img.shields.io/badge/Demo_on_OpenDataLab-blue?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTM0IiBoZWlnaHQ9IjEzNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMTIyLDljMCw1LTQsOS05LDlzLTktNC05LTksNC05LDktOSw5LDQsOSw5eiIgZmlsbD0idXJsKCNhKSIvPjxwYXRoIGQ9Im0xMjIsOWMwLDUtNCw5LTksOXMtOS00LTktOSw0LTksOS05LDksNCw5LDl6IiBmaWxsPSIjMDEwMTAxIi8+PHBhdGggZD0ibTkxLDE4YzAsNS00LDktOSw5cy05LTQtOS05LDQtOSw5LTksOSw0LDksOXoiIGZpbGw9InVybCgjYikiLz48cGF0aCBkPSJtOTEsMThjMCw1LTQsOS05LDlzLTktNC05LTksNC05LDktOSw5LDQsOSw5eiIgZmlsbD0iIzAxMDEwMSIvPjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJtMzksNjJjMCwxNiw4LDMwLDIwLDM4LDctNiwxMi0xNiwxMi0yNlY0OWMwLTQsMy03LDYtOGw0Ni0xMmM1LTEsMTEsMywxMSw4djMxYzAsMzctMzAsNjYtNjYsNjYtMzcsMC02Ni0zMC02Ni02NlY0NmMwLTQsMy03LDYtOGwyMC02YzUtMSwxMSwzLDExLDh2MjF6bS0yOSw2YzAsMTYsNiwzMCwxNyw0MCwzLDEsNSwxLDgsMSw1LDAsMTAtMSwxNS0zQzM3LDk1LDI5LDc5LDI5LDYyVjQybC0xOSw1djIweiIgZmlsbD0idXJsKCNjKSIvPjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJtMzksNjJjMCwxNiw4LDMwLDIwLDM4LDctNiwxMi0xNiwxMi0yNlY0OWMwLTQsMy03LDYtOGw0Ni0xMmM1LTEsMTEsMywxMSw4djMxYzAsMzctMzAsNjYtNjYsNjYtMzcsMC02Ni0zMC02Ni02NlY0NmMwLTQsMy03LDYtOGwyMC02YzUtMSwxMSwzLDExLDh2MjF6bS0yOSw2YzAsMTYsNiwzMCwxNyw0MCwzLDEsNSwxLDgsMSw1LDAsMTAtMSwxNS0zQzM3LDk1LDI5LDc5LDI5LDYyVjQybC0xOSw1djIweiIgZmlsbD0iIzAxMDEwMSIvPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0iYSIgeDE9Ijg0IiB5MT0iNDEiIHgyPSI3NSIgeTI9IjEyMCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIHN0b3AtY29sb3I9IiNmZmYiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyZTJlMmUiLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDE9Ijg0IiB5MT0iNDEiIHgyPSI3NSIgeTI9IjEyMCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIHN0b3AtY29sb3I9IiNmZmYiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyZTJlMmUiLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iYyIgeDE9Ijg0IiB5MT0iNDEiIHgyPSI3NSIgeTI9IjEyMCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIHN0b3AtY29sb3I9IiNmZmYiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyZTJlMmUiLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48L3N2Zz4=&labelColor=white)](https://mineru.net/OpenSourceTools/Extractor?source=github)
[![HuggingFace](https://img.shields.io/badge/Demo_on_HuggingFace-yellow.svg?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/spaces/opendatalab/MinerU)
[![ModelScope](https://img.shields.io/badge/Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://www.modelscope.cn/studios/OpenDataLab/MinerU)

### Quick CPU Demo

#### 1. Install magic-pdf

```bash
conda create -n mineru 'python=3.12' -y
conda activate mineru
pip install -U "magic-pdf[full]"
```

#### 2. Download model weight files

Refer to [How to Download Model Files](docs/how_to_download_models_en.md) for detailed instructions.

#### 3. Modify the Configuration File for Additional Configuration

After completing the [2. Download model weight files](#2-download-model-weight-files) step, the script will
automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
You can find the `magic-pdf.json` file in your 【user directory】.

> [!TIP]
> The user directory for Windows is "C:\\Users\\username", for Linux it is "/home/username", and for macOS it is "
> /Users/username".

You can modify certain configurations in this file to enable or disable features, such as table recognition:


> [!NOTE]
> If the following items are not present in the JSON, please manually add the required items and remove the comment
> content (standard JSON does not support comments).

```json
{
  // other config
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "formula-config": {
    "mfd_model": "yolo_v8_mfd",
    "mfr_model": "unimernet_small",
    "enable": true
    // The formula recognition feature is enabled by default. If you need to disable it, please change the value here to "false".
  },
  "table-config": {
    "model": "rapid_table",
    "sub_model": "slanet_plus",
    "enable": true,
    // The table recognition feature is enabled by default. If you need to disable it, please change the value here to "false".
    "max_time": 400
  }
}
```

### Using GPU

If your device supports CUDA and meets the GPU requirements of the mainline environment, you can use GPU acceleration.
Please select the appropriate guide based on your system:

- [Ubuntu 22.04 LTS + GPU](docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](docs/README_Windows_CUDA_Acceleration_en_US.md)
- Quick Deployment with Docker

> [!IMPORTANT]
> Docker requires a GPU with at least 6GB of VRAM, and all acceleration features are enabled by default.
>
> Before running this Docker, you can use the following command to check if your device supports CUDA acceleration on
> Docker.
>
> ```bash
> docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
> ```

  ```bash
  wget https://github.com/opendatalab/MinerU/raw/master/docker/global/Dockerfile -O Dockerfile
  docker build -t mineru:latest .
  docker run -it --name mineru --gpus=all mineru:latest /bin/bash -c "echo 'source /opt/mineru_venv/bin/activate' >> ~/.bashrc && exec bash"
  magic-pdf --help
  ```

### Using NPU

If your device has NPU acceleration hardware, you can follow the tutorial below to use NPU acceleration:

[Ascend NPU Acceleration](docs/README_Ascend_NPU_Acceleration_zh_CN.md)

### Using MPS

If your device uses Apple silicon chips, you can enable MPS acceleration for your tasks.

You can enable MPS acceleration by setting the `device-mode` parameter to `mps` in the `magic-pdf.json` configuration
file.

```json
{
  // other config
  "device-mode": "mps"
}
```

## Usage

### Command Line

[Using MinerU via Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)

> [!TIP]
> For more information about the output files, please refer to the [Output File Description](docs/output_file_en_us.md).

### API

[Using MinerU via Python API](https://mineru.readthedocs.io/en/latest/user_guide/usage/api.html)

### Deploy Derived Projects

Derived projects include secondary development projects based on MinerU by project developers and community
developers,  
such as application interfaces based on Gradio, RAG based on llama, web demos similar to the official website,
lightweight multi-GPU load balancing client/server ends, etc.
These projects may offer more features and a better user experience.  
For specific deployment methods, please refer to the [Derived Project README](projects/README.md)

### Development Guide

TODO

# TODO

- [x] Reading order based on the model
- [x] Recognition of `index` and `list` in the main text
- [x] Table recognition
- [x] Heading Classification
- [ ] Code block recognition in the main text
- [ ] [Chemical formula recognition](docs/chemical_knowledge_introduction/introduction.pdf)
- [ ] Geometric shape recognition