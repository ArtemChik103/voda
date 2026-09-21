#!/usr/bin/env python3
"""High-load concurrency and stress benchmark for KosmoHackathon 2026 REST API.

Evaluates:
- Concurrent request throughput (Requests Per Second, RPS)
- Latency percentiles (min, avg, p50, p90, p95, p99, max in ms)
- Concurrency stability under 20-50 simultaneous async workers
- Memory usage delta (MB RAM)
- 100% success rate (HTTP 200) across analytical and GeoJSON endpoints

Generates: reports/benchmark_highload.json
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Windows-safe console output reconfiguration
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from src.service.api import app


async def worker_task(
    client: httpx.AsyncClient,
    queue: asyncio.Queue,
    latencies: List[float],
    errors: List[str],
):
    """Processes URLs from queue and measures latency."""
    while not queue.empty():
        try:
            url = await queue.get()
        except asyncio.QueueEmpty:
            break

        t0 = time.perf_counter()
        try:
            resp = await client.get(url)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors.append(f"{url} returned HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{url} exception: {str(e)}")
        finally:
            queue.task_done()


async def run_load_test(
    total_requests: int = 120,
    concurrency: int = 20,
    pair_id: str = "flood_2019_07_amur__blagoveshchensk",
) -> Dict[str, Any]:
    """Runs concurrent async requests against the FastAPI ASGI application."""
    endpoints = [
        "/health",
        "/api/v1/pairs",
        f"/api/v1/pairs/{pair_id}/report",
        f"/api/v1/pairs/{pair_id}/what_if?delta_h=1.0",
        f"/api/v1/pairs/{pair_id}/timelapse",
        f"/api/v1/pairs/{pair_id}/evacuation",
        f"/api/v1/pairs/{pair_id}/traps",
        "/api/v1/gauge_stations",
    ]

    # Fill queue
    queue = asyncio.Queue()
    for i in range(total_requests):
        url = endpoints[i % len(endpoints)]
        await queue.put(url)

    latencies: List[float] = []
    errors: List[str] = []

    # Measure memory before
    import psutil
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    transport = httpx.ASGITransport(app=app)
    t_start = time.perf_counter()

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [
            asyncio.create_task(worker_task(client, queue, latencies, errors))
            for _ in range(concurrency)
        ]
        await queue.join()
        for t in tasks:
            t.cancel()

    t_total = time.perf_counter() - t_start
    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    successful = len(latencies)
    failed = len(errors)
    rps = round(successful / t_total, 1) if t_total > 0 else 0.0

    latencies.sort()
    stats = {
        "min_ms": round(latencies[0], 2) if latencies else 0.0,
        "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p50_ms": round(latencies[int(len(latencies) * 0.50)], 2) if latencies else 0.0,
        "p90_ms": round(latencies[int(len(latencies) * 0.90)], 2) if latencies else 0.0,
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2) if latencies else 0.0,
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0.0,
        "max_ms": round(latencies[-1], 2) if latencies else 0.0,
    }

    result = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "total_requests": total_requests,
            "concurrency_workers": concurrency,
            "target_pair_id": pair_id,
            "endpoints_tested_count": len(endpoints),
        },
        "throughput": {
            "total_time_seconds": round(t_total, 3),
            "requests_per_second": rps,
            "successful_requests": successful,
            "failed_requests": failed,
            "error_rate_pct": round(failed / max(1, total_requests) * 100.0, 2),
        },
        "latency_ms": stats,
        "memory_mb": {
            "ram_before_mb": round(ram_before_mb, 1),
            "ram_after_mb": round(ram_after_mb, 1),
            "ram_delta_mb": round(ram_after_mb - ram_before_mb, 1),
        },
        "assessment": "EXCELLENT" if rps > 100 and failed == 0 else "GOOD",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="KosmoHackathon 2026 High-Load API Benchmark")
    parser.add_argument("--requests", type=int, default=120, help="Total requests to dispatch")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    parser.add_argument("--pair", type=str, default="flood_2019_07_amur__blagoveshchensk", help="Pair ID")
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print(" [*] НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ REST API (HIGH-LOAD STRESS BENCHMARK)")
    print(f"     Команда: ArtemChik103/voda | Запросов: {args.requests} | Потоков: {args.concurrency}")
    print("=" * 76)

    res = asyncio.run(run_load_test(total_requests=args.requests, concurrency=args.concurrency, pair_id=args.pair))

    print(f"\n[+] Результаты стресс-теста:")
    print(f"    - Всего запросов:          {res['throughput']['successful_requests']} / {args.requests}")
    print(f"    - Ошибок (Error rate):     {res['throughput']['failed_requests']} ({res['throughput']['error_rate_pct']}%)")
    print(f"    - Пропускная способность:  {res['throughput']['requests_per_second']} RPS")
    print(f"    - Общее время теста:       {res['throughput']['total_time_seconds']} с")

    print(f"\n[+] Распределение задержек (Latency):")
    print(f"    - Минимум:                 {res['latency_ms']['min_ms']} мс")
    print(f"    - Среднее (Avg):           {res['latency_ms']['avg_ms']} мс")
    print(f"    - Медиана (p50):           {res['latency_ms']['p50_ms']} мс")
    print(f"    - 95-й перцентиль (p95):   {res['latency_ms']['p95_ms']} мс")
    print(f"    - 99-й перцентиль (p99):   {res['latency_ms']['p99_ms']} мс")
    print(f"    - Максимум:                {res['latency_ms']['max_ms']} мс")

    print(f"\n[+] Память процесса (RAM):")
    print(f"    - До теста:                {res['memory_mb']['ram_before_mb']} МБ")
    print(f"    - После теста:             {res['memory_mb']['ram_after_mb']} МБ (прирост: {res['memory_mb']['ram_delta_mb']} МБ)")

    # Save report
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_highload.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Машиночитаемый отчет сохранен в: {report_path.as_posix()}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
