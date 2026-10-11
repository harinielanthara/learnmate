"""Run with the backend up:  python smoke_phase3.py          (fast, no LLM needed)
                            python smoke_phase3.py --extract (also runs real concept extraction via Ollama)"""
import random
import sys
import requests

API = "http://localhost:8000"
PW = "Passw0rd!123"
results = []


def check(name, cond):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name)


def signup(tag):
    email = f"{tag}{random.randint(10**6, 10**7)}@test.dev"
    assert requests.post(f"{API}/auth/register", json={"email": email, "password": PW}).status_code == 201
    r = requests.post(f"{API}/auth/login", data={"username": email, "password": PW})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


A, B = signup("alice"), signup("bob")
sid = requests.post(f"{API}/subjects", json={"name": "ML"}, headers=A).json()["id"]


def mk(name, h=A):
    return requests.post(f"{API}/concepts", json={"subject_id": sid, "name": name}, headers=h)


# ---------------- concepts: manual CRUD
gd, lf, dv = (mk(n).json() for n in ("Gradient Descent", "Loss Function", "Derivative"))
check("A creates 3 concepts (manual => verified)", all(c["verified"] and c["source"] == "manual" for c in (gd, lf, dv)))
check("duplicate name (case/plural-insensitive) -> 409", mk("gradient descents").status_code == 409)
check("generic name rejected -> 400", mk("Introduction").status_code == 400)
check("B cannot create in A's subject", mk("Hack", B).status_code == 404)


def pre(c, p, h=A):
    return requests.post(f"{API}/concepts/{c['id']}/prerequisites", json={"prerequisite_id": p["id"]}, headers=h)


check("add edge GD requires LF", pre(gd, lf).status_code == 201)
check("add edge LF requires DV", pre(lf, dv).status_code == 201)
check("cycle DV requires GD -> 409", pre(dv, gd).status_code == 409)
check("self edge -> 400", pre(gd, gd).status_code == 400)
lst = requests.get(f"{API}/concepts", params={"subject_id": sid}, headers=A).json()
g = next(c for c in lst if c["id"] == gd["id"])
check("list shows prerequisite + dependents", [p["name"] for p in g["prerequisites"]] == ["Loss Function"]
      and next(c for c in lst if c["id"] == lf["id"])["dependent_count"] == 1)

r = requests.delete(f"{API}/concepts/{gd['id']}/prerequisites/{lf['id']}", headers=A)
check("remove edge", r.ok and r.json()["prerequisites"] == [])
check("re-adding a removed edge works", pre(gd, lf).status_code == 201)
check("verify edge", requests.patch(f"{API}/concepts/{gd['id']}/prerequisites/{lf['id']}",
                                    json={"verified": True}, headers=A).ok)

sl = mk("Slope").json()
m = requests.post(f"{API}/concepts/{sl['id']}/merge", json={"into_id": dv["id"]}, headers=A)
check("merge Slope into Derivative", m.ok)
names = [c["name"] for c in requests.get(f"{API}/concepts", params={"subject_id": sid}, headers=A).json()]
check("merged concept hidden", "Slope" not in names and "Derivative" in names)
check("rename ok / clash -> 409",
      requests.patch(f"{API}/concepts/{dv['id']}", json={"name": "Derivatives (calculus)"}, headers=A).ok
      and requests.patch(f"{API}/concepts/{lf['id']}", json={"name": "Gradient descent"}, headers=A).status_code == 409)

# ---------------- isolation
check("B cannot list A's concepts", requests.get(f"{API}/concepts", params={"subject_id": sid}, headers=B).status_code == 404)
check("B cannot edit A's concept", requests.patch(f"{API}/concepts/{gd['id']}", json={"verified": False}, headers=B).status_code == 404)
check("B cannot delete A's concept", requests.delete(f"{API}/concepts/{gd['id']}", headers=B).status_code == 404)
check("B cannot add edge on A's concept", pre(gd, lf, B).status_code == 404)
check("delete concept (tombstone)", requests.delete(f"{API}/concepts/{lf['id']}", headers=A).status_code == 204)

# ---------------- conversations (no LLM needed: empty subject + strict mode => refusal)
r = requests.post(f"{API}/qa/ask", headers=A, json={"question": "What is gradient descent?",
                                                    "subject_id": sid, "mode": "material"})
check("ask creates a saved conversation", r.ok and r.json()["conversation_id"])
cid = r.json()["conversation_id"]
r2 = requests.post(f"{API}/qa/ask", headers=A, json={"question": "And a second question?", "subject_id": sid,
                                                      "mode": "material", "conversation_id": cid})
check("follow-up stays in same conversation", r2.ok and r2.json()["conversation_id"] == cid)
convs = requests.get(f"{API}/conversations", headers=A).json()
check("A sees 1 conversation; B sees none", len(convs) == 1 and requests.get(f"{API}/conversations", headers=B).json() == [])
msgs = requests.get(f"{API}/conversations/{cid}/messages", headers=A).json()
check("4 messages saved in order", [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"])
check("B cannot read/ask into/delete A's conversation",
      requests.get(f"{API}/conversations/{cid}/messages", headers=B).status_code == 404
      and requests.post(f"{API}/qa/ask", headers=B, json={"question": "hello there", "mode": "general",
                                                          "conversation_id": cid}).status_code == 404
      and requests.delete(f"{API}/conversations/{cid}", headers=B).status_code == 404)
check("rename conversation", requests.patch(f"{API}/conversations/{cid}", json={"title": "My chat"}, headers=A).json()["title"] == "My chat")
check("delete conversation", requests.delete(f"{API}/conversations/{cid}", headers=A).status_code == 204
      and requests.get(f"{API}/conversations", headers=A).json() == [])

# ---------------- optional: real extraction with your local LLM
if "--extract" in sys.argv:
    TEXT = ("Gradient descent is an optimization algorithm. It repeatedly updates model parameters in the "
            "direction that reduces the loss function. The loss function measures how far predictions are "
            "from the true values. To know the direction, gradient descent uses the derivative of the loss "
            "function with respect to each parameter. ") * 3
    sid2 = requests.post(f"{API}/subjects", json={"name": "Extraction test"}, headers=A).json()["id"]
    up = requests.post(f"{API}/documents", headers=A, data={"subject_id": sid2},
                       files={"file": ("ml.txt", TEXT.encode())}, timeout=300)
    check("upload for extraction", up.ok and up.json()["status"] == "ready")
    x = requests.post(f"{API}/concepts/extract", headers=A, json={"subject_id": sid2}, timeout=900)
    check("extraction runs", x.ok)
    if x.ok:
        print("   ", x.json())
        got = requests.get(f"{API}/concepts", params={"subject_id": sid2}, headers=A).json()
        check("found >= 1 concept, all AI-labelled & unverified",
              len(got) >= 1 and all(c["source"] == "llm" and not c["verified"] for c in got))
        print("    concepts:", [c["name"] for c in got])
        check("AI edges are unverified", all(not p["verified"] for c in got for p in c["prerequisites"]))
        again = requests.post(f"{API}/concepts/extract", headers=A, json={"subject_id": sid2}, timeout=900).json()
        check("re-running does not duplicate concepts", again["new_concepts"] == 0)

for h in (A, B):
    requests.delete(f"{API}/auth/me", headers=h)
print(f"\n{sum(results)}/{len(results)} checks passed")
