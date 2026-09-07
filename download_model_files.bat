@echo off
setlocal

set MODEL_DIR=models\124M
set BASE_URL=https://openaipublic.blob.core.windows.net/gpt-2/models/124M

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo Downloading GPT-2 124M model files into %MODEL_DIR%...

for %%f in (
    checkpoint
    encoder.json
    hparams.json
    model.ckpt.data-00000-of-00001
    model.ckpt.index
    model.ckpt.meta
    vocab.bpe
) do (
    echo Fetching %%f...
    curl --progress-bar -o "%MODEL_DIR%\%%f" "%BASE_URL%/%%f"
    if errorlevel 1 (
        echo ERROR: Failed to download %%f
        exit /b 1
    )
)

echo.
echo Done. Model files are in %MODEL_DIR%\
