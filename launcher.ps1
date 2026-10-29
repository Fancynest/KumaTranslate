$scriptDir = "d:\HQ-Gabriel\Code\manga_translate"
Set-Location $scriptDir

# 1. Cek apakah server sudah aktif di port 7860
$active = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue

if (-not $active) {
    # Jalankan server FastAPI secara background tanpa jendela hitam
    Start-Process -FilePath "$scriptDir\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory $scriptDir -WindowStyle Hidden
    
    # Tunggu sampai server siap merespon
    $attempts = 0
    while ($attempts -lt 40) {
        Start-Sleep -Milliseconds 800
        $conn = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
        if ($conn) { break }
        $attempts++
    }
}

# 2. Buka browser otomatis
Start-Process "http://127.0.0.1:7860"
