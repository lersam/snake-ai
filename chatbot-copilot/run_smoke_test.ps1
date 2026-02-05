# Simple smoke test for the chatbot app (PowerShell)
# Starts uvicorn and performs a couple of simple HTTP calls.

$port = 8000
$proc = Start-Process -FilePath python -ArgumentList "-m uvicorn chatbot.app:app --port $port" -NoNewWindow -PassThru
Start-Sleep -Seconds 2
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/" -Method GET
    Write-Host "GET / OK"
    $body = @{question = "Hello"} | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/chat/GameProgrammingBooks/query" -Method POST -Body $body -ContentType "application/json"
    Write-Host "POST /chat/GameProgrammingBooks/query ->" $resp
} catch {
    Write-Host "Smoke test failed: $_"
} finally {
    # try to stop uvicorn
    if ($proc -and -not $proc.HasExited) { $proc.Kill() }
}
