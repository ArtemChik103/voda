@echo off
chcp 65001 > nul
echo ========================================================
echo  Запуск геопортала «Вода-Космос» (КосмоХакатон 2026)
echo  Интерфейс: http://localhost:8000
echo  Документация Swagger: http://localhost:8000/docs
echo ========================================================
python -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload
pause
