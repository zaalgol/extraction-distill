@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_env.ps1" > "%~dp0..\env_check.txt" 2>&1
