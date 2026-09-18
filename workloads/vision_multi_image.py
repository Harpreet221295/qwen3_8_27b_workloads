"""Two/three images in one request."""
from qwen_workloads import QwenClient, Workload, to_data_url, make_shapes_image, make_bar_chart
c = QwenClient(); w = Workload("vision_multi_image")

def which_has_more():
    a, ta = make_shapes_image(1); b, tb = make_shapes_image(2)
    na = ta["circles"] + ta["squares"] + ta["triangles"]; nb = tb["circles"] + tb["squares"] + tb["triangles"]
    if na == nb: return {"ok": None, "note": "equal counts, skipped"}
    r = c.ask_image("Image 1 is first, image 2 is second. Which image has more shapes? Answer '1' or '2' only.",
                    [to_data_url(a), to_data_url(b)], max_tokens=5)
    exp = "1" if na > nb else "2"
    return {"ok": exp in r.content, "output": r.content, "note": f"counts {na} vs {nb}", "latency_s": r.latency_s}

def chart_vs_chart():
    c1 = make_bar_chart({"A": 10, "B": 20, "C": 30}, "Chart one"); c2 = make_bar_chart({"A": 50, "B": 20, "C": 5}, "Chart two")
    r = c.ask_image("In which chart is bar A the tallest bar? Answer 'one' or 'two'.", [to_data_url(c1), to_data_url(c2)], max_tokens=5)
    return {"ok": "two" in r.content.lower(), "output": r.content, "latency_s": r.latency_s}

w.case("which_has_more", which_has_more).case("chart_vs_chart", chart_vs_chart)
if __name__ == "__main__": w.run()
