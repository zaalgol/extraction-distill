# Run scripts/jobs/<name>.sh in WSL from a detached, hidden wsl.exe. A job started with
# nohup from a short-lived `wsl.exe -e` call can die when WSL ends the session.
param([Parameter(Mandatory)][string]$Name)
Start-Process -WindowStyle Hidden wsl.exe -ArgumentList "-e bash /mnt/c/code/extraction-distill/scripts/jobs/$Name.sh"
