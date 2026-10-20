# Environment check for extraction-distill: GPU, driver, CUDA, WSL2, disk, RAM.
$env:WSL_UTF8 = "1"
function Section($t) { "`n===== $t =====" }

Section "Windows"
(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber | Format-List | Out-String).Trim()
"RAM GB: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } |
  ForEach-Object { "{0}: free {1:N0} GB of {2:N0} GB" -f $_.Name, ($_.Free/1GB), (($_.Used+$_.Free)/1GB) }

Section "nvidia-smi (Windows)"
try { nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv; nvidia-smi | Select-Object -First 6 } catch { "nvidia-smi not found: $_" }

Section "Tools on Windows"
foreach ($c in "ollama","uv","python","git","docker") {
  $cmd = Get-Command $c -ErrorAction SilentlyContinue
  if ($cmd) { "${c} -> $($cmd.Source)"; try { & $c --version 2>&1 | Select-Object -First 1 } catch { "  (version failed: $_)" } } else { "${c}: not found" }
}

Section "WSL"
try { wsl --version 2>&1 } catch { "wsl --version failed: $_" }
try { wsl -l -v 2>&1 } catch { "wsl -l failed: $_" }

Section "Inside default WSL distro"
try {
  wsl -e bash -lc 'echo "distro: $(. /etc/os-release; echo $PRETTY_NAME)"; uname -r; echo "nproc: $(nproc)"; free -g | head -2; df -h ~ | tail -1; echo "--- nvidia-smi in WSL"; (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "no nvidia-smi in WSL"); ls -la /usr/lib/wsl/lib/libcuda.so* 2>/dev/null | head -2; echo "--- python"; (python3 --version || echo "no python3"); (which pip3 uv nvcc 2>/dev/null || true); echo "--- docker"; (docker --version 2>/dev/null || echo "no docker in WSL")' 2>&1
} catch { "WSL bash failed: $_" }
"`nDONE"
