import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import AsyncClient
from fastapi import status
from httpx import ASGITransport

from main import app


async def main():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # get list
        r = await client.get("/api/v1/test-cases/")
        r.raise_for_status()
        items = r.json()
        if not items:
            print("No test cases found to copy")
            return
        base = items[0]
        base_id = base["id"]
        print(f"Base id: {base_id}, base priority: {base.get('priority')}")

        # Try two copies: one with priority=high, one with priority=low
        for pr in ["high", "low", "critical", "medium"]:
            payload = {
                "base_test_case_id": base_id,
                "title": f"{base['title']} - Copy ({pr})",
                "priority": pr,
            }
            resp = await client.post(f"/api/v1/test-cases/{base_id}/save-as-new", json=payload)
            print(f"POST save-as-new {pr}: status {resp.status_code}")
            if resp.status_code != status.HTTP_200_OK:
                print(resp.text)
                continue
            data = resp.json()
            new_tc = data.get("test_case", {})
            print(f" -> returned priority: {new_tc.get('priority')} (id={new_tc.get('id')})")

if __name__ == "__main__":
    asyncio.run(main())
