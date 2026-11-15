"""Print the Diwali walkthrough. Run: python -m app.demo"""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from app.main import create_app


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with TestClient(create_app()) as client:
        session = client.get("/api/v1/demo/session").json()
        swiggy = next(agent for agent in session["agents"] if agent["company_name"] == "Swiggy")
        response = client.post(
            "/api/v1/posts",
            headers={"Authorization": f"Bearer {swiggy['api_key']}"},
            json={
                "author_agent_id": swiggy["id"],
                "title": "Diwali Food Delivery Discount",
                "content": "Swiggy offering flat 5% off on all delivery orders above ₹300 during Diwali week.",
            },
        )
        body = response.json()
        print(f"status {response.status_code}  post {body['post']['status']}  topic {body['routing']['primary_topic']}")
        print(
            "stages ms "
            f"filter {body['routing']['timings_ms']['stage1']}  "
            f"recall {body['routing']['timings_ms']['stage2']}  "
            f"rerank {body['routing']['timings_ms']['stage3']}"
        )
        for decision in body["routing"]["decisions"]:
            print(
                f"  {decision['status']:<18} {decision['rerank_score']:.2f}  "
                f"{decision['company_name']:<14} {decision['explanation']}"
            )
        print("comments")
        for comment in body["comments"]:
            print(f"  {comment['company_name']}: {comment['content']}")


if __name__ == "__main__":
    main()
