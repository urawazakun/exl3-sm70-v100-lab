#!/usr/bin/env python3
"""Needle probe v3 - fixed for reasoning models.

v2's bug: max_tokens=64 was far too small for a thinking model, so every probe returned
finish_reason='length' with content='' (all 64 tokens went to reasoning_content) and was scored MISS.
That is a harness artifact, not a context-quality failure.

v3 changes:
  * default max_tokens = 1024 (enough for reasoning + the answer),
  * the expected code is searched in BOTH `content` and `reasoning_content`,
  * the raw head of both channels is printed so a miss can be diagnosed.

Usage: probe3.py <base_url> <expected_code> <prompt_json> <out_prefix> [max_tokens]
"""
import json, sys, urllib.request, urllib.error, time

def post(url, data_bytes, timeout):
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode("utf-8", "replace")[:400]}
    except Exception as e:
        return {"error": "exception", "body": str(e)[:400]}

def main():
    base = sys.argv[1].rstrip("/")
    code = sys.argv[2]
    pj = sys.argv[3]
    out_prefix = sys.argv[4]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 1024

    payload = json.load(open(pj, encoding="utf-8"))
    payload["max_tokens"] = max_tokens
    payload["temperature"] = 0.0
    payload["stream"] = False

    t0 = time.time()
    resp = post(base + "/v1/chat/completions", json.dumps(payload).encode(), 900)
    wall = time.time() - t0
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=1)

    if "error" in resp:
        print("RESULT code=%s hit=ERROR detail=%r wall=%.1fs" % (code, resp, wall))
        return

    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = (resp.get("choices") or [{}])[0].get("finish_reason")
    tm = resp.get("timings") or {}

    hit = code in content or code in reasoning
    where = "content" if code in content else ("reasoning" if code in reasoning else "none")
    print("RESULT code=%s hit=%s where=%s finish=%s wall=%.1fs prompt_n=%s prompt_ms=%s prompt_tps=%s "
          "predicted_n=%s predicted_tps=%s"
          % (code, "HIT" if hit else "MISS", where, finish, wall,
             tm.get("prompt_n"), round(tm.get("prompt_ms", 0), 1), round(tm.get("prompt_per_second", 0), 2),
             tm.get("predicted_n"), round(tm.get("predicted_per_second", 0), 2)))
    print("   content[:160]=%r" % content[:160].replace("\n", " "))
    print("   reasoning[:160]=%r" % reasoning[:160].replace("\n", " "))

if __name__ == "__main__":
    main()
