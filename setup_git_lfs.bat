@echo off
echo Setting up Git LFS for Hugging Face deployment...
echo.

REM Check if Git LFS is installed
git lfs version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git LFS is not installed!
    echo.
    echo Please install it from: https://git-lfs.github.com/
    echo.
    echo After installing, run this script again.
    pause
    exit /b 1
)

echo Git LFS is installed
echo.

REM Initialize Git LFS
echo Initializing Git LFS...
git lfs install

REM Track model files
echo Tracking large model files...
git lfs track "models/*.pkl"
git lfs track "models/*.pth"
git lfs track "models/*.parquet"

REM Add .gitattributes
echo Adding .gitattributes...
git add .gitattributes

REM Commit
echo Committing LFS configuration...
git commit -m "Configure Git LFS for model files"

echo.
echo ==========================================
echo Git LFS setup complete!
echo ==========================================
echo.
echo Now you can push to Hugging Face:
echo.
echo $env:HF_TOKEN = "YOUR_TOKEN"
echo git push https://morriase:$env:HF_TOKEN@huggingface.co/spaces/morriase/forex-live_server main --force
echo.
pause
