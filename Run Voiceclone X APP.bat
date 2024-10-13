@echo off

REM Change directory to the Modules folder inside Voiceclone X (App)
cd /d "%~dp0\Modules"

REM Execute the create_and_activate_environment Python script
python main_setup_module.py

REM Pause to view any output
pause
