"""Resumable Anthropic Message Batches driver. Checkpoints batch ids; survives restarts.

One batch = up to 100k requests, async, 50% cheaper. Use for inventory design (1 req/word)
and sense tagging (1 req per word-chunk of occurrences).
"""
import json, os, time, glob
import anthropic

# load key from the lexicon .env
for ln in open("/home/remy/Code/lexicon/.env"):
    if ln.startswith("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = ln.split("=",1)[1].strip().strip('"').strip("'")
CLIENT = anthropic.Anthropic()
MAXREQ = 12000
CKPT_DIR = "out/batches"
os.makedirs(CKPT_DIR, exist_ok=True)

def submit(requests, tag):
    """requests: list of dict(custom_id, system, user, model, max_tokens). Returns batch ids."""
    ids = []
    for b in range(0, len(requests), MAXREQ):
        chunk = requests[b:b+MAXREQ]
        batch = CLIENT.messages.batches.create(requests=[{
            "custom_id": r["custom_id"],
            "params": {"model": r.get("model","claude-sonnet-5"),
                       "max_tokens": r.get("max_tokens",1500),
                       **({"thinking":{"type":"disabled"}} if r.get("thinking_off") else {}),
                       "system": r["system"],
                       "messages": [{"role":"user","content": r["user"]}]}
        } for r in chunk])
        ids.append(batch.id)
        print(f"  submitted batch {batch.id}  ({len(chunk)} reqs)", flush=True)
    json.dump(ids, open(f"{CKPT_DIR}/{tag}.json","w"))
    return ids

def poll(tag, interval=30):
    ids = json.load(open(f"{CKPT_DIR}/{tag}.json"))
    while True:
        st = [CLIENT.messages.batches.retrieve(i) for i in ids]
        done = sum(1 for s in st if s.processing_status == "ended")
        counts = {}
        for s in st:
            for k,v in dict(s.request_counts).items(): counts[k]=counts.get(k,0)+v
        print(f"  {tag}: {done}/{len(ids)} batches ended  {counts}", flush=True)
        if done == len(ids): return ids
        time.sleep(interval)

def collect(tag):
    ids = json.load(open(f"{CKPT_DIR}/{tag}.json"))
    out = {}
    for i in ids:
        for res in CLIENT.messages.batches.results(i):
            if res.result.type == "succeeded":
                txt = next((b.text for b in res.result.message.content
                            if getattr(b,"type",None)=="text"), None)
                out[res.custom_id] = txt
            else:
                out[res.custom_id] = None
    return out
