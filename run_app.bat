@echo off
echo Installing Python packages from requirements.txt...
pip install -r requirements.txt

if exist packages.txt (
    echo Installing system packages from packages.txt...
    for /f "tokens=*" %%i in (packages.txt) do (
        echo Installing %%i...
        winget install --id %%i -e --accept-package-agreements --accept-source-agreements
    )
) else (
    echo packages.txt not found, skipping system package installation.
)

echo Starting the application...
start "" /B python app.py --port 7860 > app.log 2>&1

echo Application is running in the background.
echo Check app.log for logs.
echo To stop the application, run: taskkill /f /im python.exe
