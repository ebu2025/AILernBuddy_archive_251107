@echo off
setlocal

set "VENV_ACTIVATE=.venv\Scripts\activate.bat"

if not exist "%VENV_ACTIVATE%" (
    echo [INFO] Virtuelle Umgebung nicht gefunden. Erstelle .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Konnte virtuelle Umgebung nicht erstellen.
        exit /b 1
    )
)

call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Konnte virtuelle Umgebung nicht aktivieren.
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Konnte pip nicht aktualisieren.
    exit /b 1
)

pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Konnte Abhaengigkeiten nicht installieren.
    exit /b 1
)

set GPT4ALL_URL=http://localhost:4891/v1/chat/completions
set MODEL_ID=DeepSeek-R1-Distill-Qwen-14B
set ENGINE_MODE=elo
set LLM_TIMEOUT=1800

:: WICHTIG: jetzt per max_tokens steuern (server versteht n_predict nicht)
set SEND_MAX_TOKENS=true
set MAX_TOKENS=1500
set SIMPLE_MAX_TOKENS=1500

:: optionale Feineinstellungen
set LLM_TEMPERATURE=0.2
set LLM_TOP_P=0.95
set LLM_REPEAT_PENALTY=1.1

python -m uvicorn app:app --reload --port 8000

pause