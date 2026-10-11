"""Run with the backend up:  python smoke_phase2.py
Checks login rules and that user B can never see or touch user A's data."""
import random
import requests

API = "http://localhost:8000"
PW = "Passw0rd!123"
results = []


def check(name, cond):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name)


def signup(tag):
    email = f"{tag}{random.randint(10**6, 10**7)}@test.dev"
    r = requests.post(f"{API}/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201, r.text
    r = requests.post(f"{API}/auth/login", data={"username": email, "password": PW})
    assert r.ok, r.text
    return email, {"Authorization": f"Bearer {r.json()['access_token']}"}


emailA, A = signup("alice")
emailB, B = signup("bob")

check("no token -> 401", requests.get(f"{API}/subjects").status_code == 401)
check("garbage token -> 401",
      requests.get(f"{API}/subjects", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401)
check("duplicate email -> 409",
      requests.post(f"{API}/auth/register", json={"email": emailA, "password": PW}).status_code == 409)
check("wrong password -> 401",
      requests.post(f"{API}/auth/login", data={"username": emailA, "password": "nope-nope-1"}).status_code == 401)
check("short password -> 422",
      requests.post(f"{API}/auth/register", json={"email": "x@y.com", "password": "123"}).status_code == 422)

sa = requests.post(f"{API}/subjects", json={"name": "Physics"}, headers=A)
check("A creates subject", sa.status_code == 201)
sid = sa.json()["id"]
check("A duplicate subject -> 409",
      requests.post(f"{API}/subjects", json={"name": "Physics"}, headers=A).status_code == 409)
check("B does not see A's subject", requests.get(f"{API}/subjects", headers=B).json() == [])
check("B cannot delete A's subject", requests.delete(f"{API}/subjects/{sid}", headers=B).status_code == 404)

TEXT = ("Photosynthesis is the process by which green plants convert light energy into chemical energy. "
        "It takes place in the chloroplasts, where chlorophyll absorbs sunlight. "
        "The light reactions produce ATP and NADPH, which power the Calvin cycle to build glucose "
        "from carbon dioxide. ") * 2
up = requests.post(f"{API}/documents", headers=A, data={"subject_id": sid},
                   files={"file": ("bio.txt", TEXT.encode())}, timeout=300)
check("A uploads doc into subject", up.ok and up.json()["status"] == "ready" and up.json()["subject_id"] == sid)
did = up.json()["id"]
check("B cannot upload into A's subject",
      requests.post(f"{API}/documents", headers=B, data={"subject_id": sid},
                    files={"file": ("x.txt", TEXT.encode())}).status_code == 404)
check("B does not see A's documents", requests.get(f"{API}/documents", headers=B).json() == [])
check("B cannot delete A's document", requests.delete(f"{API}/documents/{did}", headers=B).status_code == 404)
check("B cannot ask against A's subject",
      requests.post(f"{API}/qa/ask", headers=B, json={"question": "What is photosynthesis?",
                                                      "subject_id": sid}).status_code == 404)
leak = requests.post(f"{API}/qa/ask", headers=B,
                     json={"question": "What is photosynthesis?", "document_ids": [did]}).json()
check("B asking with A's document id gets no content", not leak["grounded"] and leak["citations"] == [])
prof = requests.get(f"{API}/profile", headers=A).json()
check("A profile counts", prof["subject_count"] == 1 and prof["ready_document_count"] == 1)

for h in (A, B):
    requests.delete(f"{API}/auth/me", headers=h)
check("deleted account can no longer log in",
      requests.post(f"{API}/auth/login", data={"username": emailA, "password": PW}).status_code == 401)

print(f"\n{sum(results)}/{len(results)} checks passed")
