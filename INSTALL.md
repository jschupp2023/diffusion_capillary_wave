# Installation without Conda

The analysis scripts require Python 3.10 or newer. Python 3.12 is recommended.

## Linux or macOS

From the repository directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, run a script from the repository directory, for example:

```bash
python pod_2d.py --help
```

Deactivate the environment when finished:

```bash
deactivate
```

## Video export

`capillary_video.py` additionally needs the `ffmpeg` program. It is not a
Python package and must be installed separately:

```bash
# Ubuntu or Debian
sudo apt update
sudo apt install ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

On Windows, install FFmpeg and ensure its `bin` directory is available on the
system `PATH`. Confirm the installation with:

```bash
ffmpeg -version
```

## Jupyter notebooks (optional)

To work with the included notebooks, install JupyterLab in the active virtual
environment:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name capillarywave --display-name "Python (capillarywave)"
```
