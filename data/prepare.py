"""
Download FRANK and RoSE and convert them into the SummEval JSONL layout
read by eval.NewDataset, one file per corpus:

    data/frank_cnndm.jsonl   Factuality stored as consistency
    data/frank_xsum.jsonl    Factuality stored as consistency
    data/rose_cnndm.jsonl    ACU stored as relevance

Dimensions a corpus does not annotate are 0. Candidates are lowercased
and PTB-tokenised like SummEval's, articles that also occur in SummEval
are dropped, and RoSE's `gold` system is dropped (it is the reference).
Standard library only.
"""

import io
import json
import os
import re
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FRANK = "https://raw.githubusercontent.com/artidoro/frank/main/data/"
ROSE = "https://huggingface.co/datasets/Salesforce/rose/resolve/main/rose_data.tar.gz"
DIMS = ["coherence", "consistency", "fluency", "relevance"]


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def key(text):
    return re.sub(r"\W", "", text.lower())[:200]


def ptb(s):
    s = " ".join(s.replace("\xa0", " ").lower().split())
    s = re.sub(r'([,;:!?()"])', r" \1 ", s)
    s = re.sub(r"\.(\s|$)", r" . ", s)
    s = re.sub(r"(\w)(n't)\b", r"\1 \2", s)
    s = re.sub(r"(\w)('s|'re|'ve|'ll|'d|'m)\b", r"\1 \2", s)
    return " ".join(s.split())


def write(name, docs):
    path = os.path.join(HERE, name + ".jsonl")
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"{path}: {len(docs)} articles × {len(docs[0]['machine_summaries'])} systems")


def entry(doc_id, text, ref, systems, dim):
    """systems: {name: (summary, score)}, written in sorted name order."""
    names = sorted(systems)
    out = {
        "id": doc_id,
        "text": text,
        "machine_summaries": [ptb(systems[n][0]) for n in names],
        "human_summaries": [ref],
        "systems": names,
    }
    for d in DIMS:
        out[d] = [systems[n][1] if d == dim else 0.0 for n in names]
    return out


def main():
    summeval = {
        key(json.loads(l)["text"])
        for l in open(os.path.join(HERE, "..", "pkg", "dataset", "summeval.jsonl"))
    }

    bench = json.loads(fetch(FRANK + "benchmark_data.json"))
    ann = {(a["hash"], a["model_name"]): a for a in json.loads(fetch(FRANK + "human_annotations.json"))}
    frank = {"cnndm": {}, "xsum": {}}
    for b in bench:
        a = ann[(b["hash"], b["model_name"])]
        corpus = "cnndm" if a["dataset"] == "cnndm" else "xsum"
        article = b["article"]
        if corpus == "xsum":
            article = re.sub(r"([.!?])([A-Z])", r"\1 \2", article)
        d = frank[corpus].setdefault(b["hash"], {"text": article, "ref": b["reference"], "sys": {}})
        d["sys"][b["model_name"]] = (b["summary"], float(a["Factuality"]))
    for corpus, docs in frank.items():
        write(f"frank_{corpus}", [
            entry(h, d["text"], d["ref"], d["sys"], "consistency")
            for h, d in docs.items() if key(d["text"]) not in summeval
        ])

    tar = tarfile.open(fileobj=io.BytesIO(fetch(ROSE)))
    raw = tar.extractfile("rose_data/cnndm.test.acus.aggregated.jsonl").read().decode()
    docs = []
    for line in raw.splitlines():
        r = json.loads(line)
        if key(r["source"]) in summeval:
            continue
        sys = {
            n: (s, r["annotations"][n]["acu"])
            for n, s in r["system_outputs"].items() if n != "gold"
        }
        docs.append(entry(r["example_id"], r["source"], r["reference"], sys, "relevance"))
    write("rose_cnndm", docs)


if __name__ == "__main__":
    main()
