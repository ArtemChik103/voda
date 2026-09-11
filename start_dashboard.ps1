# 1-click PowerShell launcher for Web-GIS dashboard
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Запуск геопортала «Вода-Космос» (КосмоХакатон 2026)" -ForegroundColor Green
Write-Host " Интерфейс: http://localhost:8000" -ForegroundColor Yellow
Write-Host " Документация Swagger: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

python -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload
