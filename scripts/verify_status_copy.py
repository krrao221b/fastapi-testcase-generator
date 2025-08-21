import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import AsyncClient
from httpx import ASGITransport

from main import app

async def main():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v1/test-cases/")
        r.raise_for_status()
        items = r.json()
        if not items:
            print("No test cases found to copy")
            return
        base = items[0]
        base_id = base["id"]
        print(f"Base id: {base_id}, base status: {base.get('status')}")

        # Try a few status inputs (string case variations and dict form)
        samples = ["active", "DEPRECATED", {"value": "draft"}]
        for st in samples:
            payload = {
                "base_test_case_id": base_id,
                "title": f"{base['title']} - Copy (status={st})",
                "status": st,
            }
            resp = await client.post(f"/api/v1/test-cases/{base_id}/save-as-new", json=payload)
            print(f"POST save-as-new status={st}: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                tc = data.get("test_case", {})
                print(f" -> returned status: {tc.get('status')} (id={tc.get('id')})")
            else:
                print(resp.text)

if __name__ == "__main__":
    asyncio.run(main())
