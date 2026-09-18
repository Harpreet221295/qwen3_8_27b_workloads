"""OCR: render text to an image, ask the model to transcribe it, measure char accuracy."""
import argparse, difflib
from qwen_workloads import QwenClient, Workload, to_data_url, make_text_image
c = QwenClient(); w = Workload("vision_ocr")

DOC = """INVOICE #2026-0917
Bill to: Harpreet Singh
Item: GPU rental (H100 80GB)   Qty: 12 hrs   Rate: $2.49/hr
Subtotal: $29.88
Tax (13%): $3.88
Total due: $33.76"""

def transcribe():
    r = c.ask_image("Transcribe all text in this image exactly, preserving line breaks. Output only the text.",
                    [to_data_url(make_text_image(DOC))], max_tokens=300)
    ratio = difflib.SequenceMatcher(None, DOC.strip(), r.content.strip()).ratio()
    return {"ok": ratio > 0.9, "similarity": round(ratio, 3), "output": r.content, "latency_s": r.latency_s, "note": f"sim={ratio:.2f}"}

def extract_field():
    r = c.ask_image("What is the total amount due? Reply with the dollar amount only.",
                    [to_data_url(make_text_image(DOC))], max_tokens=20)
    return {"ok": "33.76" in r.content, "output": r.content, "latency_s": r.latency_s}

def to_json():
    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}, "total_due": {"type": "number"},
              "quantity_hours": {"type": "integer"}}, "required": ["invoice_id", "total_due", "quantity_hours"]}
    r = c.ask_image("Extract invoice_id, total_due and quantity_hours as JSON.", [to_data_url(make_text_image(DOC))],
                    max_tokens=100, response_format={"type": "json_schema", "json_schema": {"name": "inv", "schema": schema}})
    import json
    try: d = json.loads(r.content); ok = d["total_due"] == 33.76 and d["quantity_hours"] == 12
    except Exception: ok = False
    return {"ok": ok, "output": r.content, "latency_s": r.latency_s}

def real(path):
    def fn():
        r = c.ask_image("Transcribe all text in this image.", [to_data_url(path)], max_tokens=800)
        return {"ok": None, "output": r.content, "latency_s": r.latency_s, "note": r.content[:100]}
    return fn

w.case("transcribe", transcribe).case("extract_field", extract_field).case("to_json", to_json)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); a = ap.parse_args()
    if a.image: w.case("real_image", real(a.image))
    w.run()
