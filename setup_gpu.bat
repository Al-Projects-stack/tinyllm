@echo off
REM =========================================================
REM  setup_gpu.bat -- Install CUDA-enabled PyTorch for GTX 1650
REM  Run this ONCE before resuming training on your GPU.
REM
REM  Your GPU  : NVIDIA GeForce GTX 1650 (Turing, sm_75)
REM  Driver    : 596.49   (supports up to CUDA 13.2)
REM  VRAM      : 4 GB
REM  dtype     : float16  (bfloat16 not supported on Turing)
REM =========================================================

echo [setup] Uninstalling CPU-only PyTorch...
pip uninstall torch -y

echo [setup] Installing PyTorch 2.x with CUDA 12.8 support...
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo [setup] Verifying CUDA is available...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found!'; print('CUDA OK --', torch.cuda.get_device_name(0))"

if %ERRORLEVEL% neq 0 (
    echo [setup] CUDA check failed. Trying CUDA 12.6 wheel instead...
    pip uninstall torch -y
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found!'; print('CUDA OK --', torch.cuda.get_device_name(0))"
)

echo.
echo [setup] Done! Resume training with:
echo   python train.py --steps 10000 --resume checkpoints\step_001500.pt
echo.
echo Note: dtype defaults to bfloat16 but will auto-downgrade to float16
echo       for your GTX 1650 (Turing). No extra flags needed.
pause
