$ErrorActionPreference = "Stop"

Write-Host "=== UniGaze Large local setup ===" -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher 'py' not found. Install Python 3.10 and enable the Python Launcher."
}

if (-not (Test-Path ".venv")) {
    py -3.10 -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-windows-cuda118.txt

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git not found. Install Git for Windows and restart PowerShell."
}

New-Item -ItemType Directory -Force -Path "third_party" | Out-Null

if (-not (Test-Path "third_party\UniGaze")) {
    git clone https://github.com/ut-vision/UniGaze.git third_party/UniGaze
} else {
    Write-Host "third_party\UniGaze already exists - clone skipped."
}

New-Item -ItemType Directory -Force -Path "input" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "1) Put a video at input\input.mp4"
Write-Host "2) In VS Code select interpreter: .venv\Scripts\python.exe"
Write-Host "3) Run: python unigaze_large_inference.py"
Write-Host ""
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
