"""Synthetic shape images with known ground truth + optional real image."""
import argparse, re
from qwen_workloads import QwenClient, Workload, to_data_url, make_shapes_image
c = QwenClient(); w = Workload("vision_describe")

def count_case(seed):
    def fn():
        img, truth = make_shapes_image(seed)
        r = c.ask_image("Count the circles, squares and triangles. Answer exactly as: circles=N squares=N triangles=N",
                        [to_data_url(img)], max_tokens=60)
        got = {k: int(v) for k, v in re.findall(r"(circles|squares|triangles)\s*=\s*(\d+)", r.content.lower())}
        exact = all(got.get(k) == truth[k] for k in ["circles", "squares", "triangles"])
        return {"ok": exact, "truth": truth, "output": r.content, "latency_s": r.latency_s,
                "note": f"truth={ {k: truth[k] for k in ['circles','squares','triangles']} } got={got}"}
    return fn

def color_case():
    img, truth = make_shapes_image(3)
    dominant = max(truth["colors"], key=truth["colors"].get)
    r = c.ask_image("Which color appears on the most shapes? One word.", [to_data_url(img)], max_tokens=10)
    return {"ok": dominant in r.content.lower(), "output": r.content, "truth": dominant, "latency_s": r.latency_s}

def free_description(path):
    def fn():
        r = c.ask_image("Describe this image in 3 sentences.", [to_data_url(path)], max_tokens=200)
        return {"ok": None, "output": r.content, "latency_s": r.latency_s, "note": r.content[:100]}
    return fn

for s in range(3): w.case(f"count_seed{s}", count_case(s))
w.case("dominant_color", color_case)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); a = ap.parse_args()
    if a.image: w.case("describe_real_image", free_description(a.image))
    w.run()
