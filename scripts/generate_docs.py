"""
scripts/generate_docs.py

Creates the SYNTHETIC test corpus and the eval question set. Everything
here is invented (fictional wells, operators, people, numbers). Same seed
-> same files, every time:

    python scripts/generate_docs.py

Writes:
  sample_docs/ddr_<well>_dayNN.md   36 new daily drilling reports (6 wells x 6 days)
  eval/eval_set.json                30 questions with reference answers + gold evidence

The 4 hand-written reports already in sample_docs/ (Condor-7 and Falcon-1)
are never touched. Questions about them are written out by hand below.

Things that make retrieval hard on purpose:
  - Condor-9 / Falcon-3 are look-alikes of Condor-7 / Falcon-1 (same operator,
    same days, similar numbers).
  - Mud weight, ROP etc. change every day within a well.
  - 10 "long" reports have an Operations Log, so they split into 2 chunks.
    The second chunk (mud, NPT, HSE) does not contain the well name.
  - Routine sentences (HSE, plans, hole conditions) repeat word for word.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "sample_docs"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.json"
SEED = 42

# name, operator, rig, company man, first day, mud, section, start depth
WELLS = [
    ("Condor-9", "Meridian Resources", "Rig 91", "R. Haldane", 3, "spud", '17 1/2"', 450),
    ("Falcon-3", "Northbrook Energy", "Rig 209", "M. Duarte", 10, "kcl", '12 1/4"', 7900),
    ("Harrier-2", "Ashgrove Petroleum", "Rig 33", "L. Brandt", 1, "spud", '17 1/2"', 400),
    ("Osprey-5", "Ashgrove Petroleum", "Rig 47", "K. Yilmaz", 7, "obm", '8 1/2"', 10200),
    ("Kestrel-4", "Tidewater Drilling Partners", "Rig 502", "S. Nakamura", 15, "kcl", '8 1/2"', 12100),
    ("Merlin-8", "Tidewater Drilling Partners", "Rig 507", "P. Oyelaran", 2, "spud", '17 1/2"', 520),
]
DAYS_PER_WELL = 6

# Reports that get a long Operations Log (so they split across 2 chunks).
LONG_DOCS = {("Falcon-3", 12), ("Condor-9", 5), ("Osprey-5", 9), ("Kestrel-4", 17),
             ("Merlin-8", 5), ("Harrier-2", 3), ("Falcon-3", 14), ("Kestrel-4", 19),
             ("Osprey-5", 10), ("Merlin-8", 6)}

# Hand-placed events so questions have known answers.
NPT_FORCED = {("Osprey-5", 8): (2.0, "a top drive hydraulic leak"),
              ("Osprey-5", 9): (1.5, "a mud pump liner failure")}
HSE_FORCED = {("Osprey-5", 9): "One first-aid case: crew member sustained a minor pinched finger "
                               "while operating the catwalk pipe handler. Treated on site, no further "
                               "medical attention required. Incident report filed and reviewed with crew."}
LOG_EVENTS = {("Merlin-8", 5): "At 15:10 lost circulation of 40 bbl was observed; an LCM pill was "
                               "pumped and full returns were restored by 16:20."}

MUD = {  # mud type line, base weight range, extra lines
    "spud": ("Spud mud, bentonite-based.", (9.0, 9.6)),
    "kcl": ("Water-based, KCl-polymer system.", (10.4, 11.4)),
    "obm": ("Oil-based, synthetic.", (12.0, 12.8)),
}
ROP_RANGE = {'17 1/2"': (18, 30), '12 1/4"': (9, 15), '8 1/2"': (6, 11)}
FORMATION = {'17 1/2"': "unconsolidated formation", '12 1/4"': "interbedded sand and shale",
             '8 1/2"': "competent shale with minor sand stringers"}
NPT_CAUSES = ["a top drive hydraulic leak", "a mud pump liner failure", "a delayed mud delivery from the supply base",
              "waiting on weather", "a draw-works brake inspection", "an MWD tool signal loss"]
CONDITIONS = ["Hole conditions stable, no losses observed.",
              "Hole conditions stable, no losses observed. Minor tight spots on connections.",
              "Circulated bottoms up before continuing to drill ahead."]
HSE_ROUTINE = ["Routine safety inspection of surface equipment completed. No incidents or near-misses reported.",
               "Pre-job safety meeting held prior to operations. No incidents or near-misses reported."]
NEXT_PLAN = ["Continue drilling ahead to planned section depth. Run wiper trip if hole conditions deteriorate.",
             "Continue drilling ahead. Circulate bottoms up and take surveys at each connection."]
LOG_LINES = ["Drilled ahead from {a} ft to {b} ft, WOB {wob} klbs, rotary {rpm} rpm, standpipe pressure {spp} psi.",
             "Made connection, took survey, circulated for {m} minutes.",
             "Pumped sweep, observed shakers, returns normal.",
             "Checked pump pressures and recorded slow-pump rates at {spp} psi.",
             "Serviced top drive and inspected drilling line."]

# Words that must NOT appear in any report: they are used by the
# "not documented" questions.
NOT_DOCUMENTED_TERMS = ["h2s", "toolpusher", "cost", "inclination"]


def fmt(n):
    return f"{n:,}"


def build_reports(rng):
    """Returns {(well, day): record dict with all facts + the markdown text}."""
    used_ids = set()

    def unique_id(prefix):
        while True:
            n = f"{prefix}-{rng.randint(10000, 99999)}"
            if n not in used_ids:
                used_ids.add(n)
                return n

    reports = {}
    for name, operator, rig, man, first_day, mud_key, section, start_depth in WELLS:
        mud_type, (w_lo, w_hi) = MUD[mud_key]
        weight = round(rng.uniform(w_lo, w_hi), 1)
        depth = start_depth
        for day in range(first_day, first_day + DAYS_PER_WELL):
            weight = round(weight + rng.choice([0.0, 0.1, 0.1, 0.2]), 1)
            if (name, day) in NPT_FORCED:
                npt_h, cause = NPT_FORCED[(name, day)]
            elif rng.random() < 0.45:
                npt_h, cause = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), rng.choice(NPT_CAUSES)
            else:
                npt_h, cause = 0, None
            rop = round(rng.uniform(*ROP_RANGE[section]), 1)
            feet = int(round(rop * (rng.randint(14, 20) - npt_h) / 5) * 5)
            start, depth = depth, depth + feet
            visc = rng.randint(34, 52)
            bit = unique_id("BS")
            wo = unique_id("WO") if cause else None
            r = dict(well=name, day=day, doc=f"ddr_{name.lower().replace('-', '')}_day{day:02d}.md",
                     operator=operator, rig=rig, man=man, mud_type=mud_type, weight=weight, rop=rop,
                     start=start, end=depth, visc=visc, npt_h=npt_h, cause=cause, wo=wo, bit=bit,
                     section=section, long=(name, day) in LONG_DOCS)

            lines = [f"# Daily Drilling Report -- {name}", "## Report Info", f"Well: {name}",
                     f"Report Date: Day {day}", f"Operator: {operator}", f"Rig: {rig}",
                     f"Report prepared by: {man}, Company Man", "",
                     "## Operations Summary",
                     f"Drilled {section} hole section from {fmt(start)} ft to {fmt(depth)} ft using {section} bit "
                     f"(bit serial number {bit}). Rate of penetration averaged {rop} ft/hr through "
                     f"{FORMATION[section]}. {rng.choice(CONDITIONS)}", ""]
            if r["long"]:
                lines += ["## Operations Log"]
                lines += operations_log(rng, start, depth, LOG_EVENTS.get((name, day)))
            lines += ["## Mud Properties", f"Mud type: {mud_type}", f"Mud weight: {weight:.1f} ppg.",
                      f"Funnel viscosity: {visc} sec/qt."]
            if mud_key == "spud":
                lines.append(f"Sand content: {rng.randint(8, 16) / 10:.1f}%.")
            else:
                chl = rng.randrange(15000, 24000, 100)
                chl = chl + 100 if chl == 19200 else chl  # 19,200 is Falcon-1 Day 13's value (keyword question)
                lines.append(f"Chlorides: {fmt(chl)} mg/L." if mud_key == "kcl" else "Oil/water ratio: 80/20.")
                lines.append(f"Solids content: {rng.randint(55, 90) / 10:.1f}%.")
            lines += ["", "## NPT and Downtime Events"]
            if cause:
                hh = f"{rng.randint(1, 23):02d}:{rng.choice(['00', '30'])}"
                lines.append(f"{npt_h:.1f} hours non-productive time recorded due to {cause}. "
                             f"Work order {wo} raised; operations resumed at {hh}.")
            else:
                lines.append("No non-productive time recorded during this reporting period.")
            lines += ["", "## HSE Notes", HSE_FORCED.get((name, day)) or rng.choice(HSE_ROUTINE), "",
                      "## Personnel on Site",
                      f"{rng.randint(11, 15)} rig crew, {rng.randint(1, 2)} mud engineers, 1 company man, "
                      f"1 directional driller.", "", "## Next 24-Hour Plan", rng.choice(NEXT_PLAN)]
            r["text"] = "\n".join(lines) + "\n"
            reports[(name, day)] = r
    return reports


def operations_log(rng, start, end, event):
    """Three paragraphs (one per tour) of timestamped log lines. Filler with
    lots of numbers, so the long reports really exceed one chunk."""
    paragraphs = []
    for tour in range(3):
        entries = []
        for i in range(6):
            t0 = tour * 8 + i * 1.3
            stamp = f"{int(t0):02d}:{int((t0 % 1) * 60):02d}"
            line = rng.choice(LOG_LINES).format(
                a=fmt(start + rng.randint(0, 50)), b=fmt(end - rng.randint(0, 50)), wob=rng.randint(10, 30),
                rpm=rng.randint(80, 140), spp=fmt(rng.randrange(2400, 3400, 10)), m=rng.randint(10, 45))
            entries.append(f"{stamp} {line}")
        paragraphs.append(" ".join(entries))
    if event:
        paragraphs[1] += " " + event
    return [p + "\n" for p in paragraphs]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def build_questions(R):
    qs = []

    def ev(doc, text):
        return {"doc": doc, "text": text}

    def add(cat, question, reference, evidence=(), **extra):
        qs.append({"id": f"q{len(qs) + 1:02d}", "category": cat, "user_input": question,
                   "reference": reference, "gold_docs": sorted({e["doc"] for e in evidence}),
                   "evidence": list(evidence), **extra})

    c7d4, c7d5 = "ddr_condor7_day04.md", "ddr_condor7_day05.md"
    f1d12, f1d13 = "ddr_falcon1_day12.md", "ddr_falcon1_day13.md"

    # --- the original hand-written questions (kept as they were) ---------
    add("single_fact", "What was the mud weight on Condor-7 on Day 4?", "9.2 ppg.", [ev(c7d4, "Mud weight: 9.2 ppg.")])
    add("single_fact", "What was the average rate of penetration on Falcon-1 Day 12?", "11.7 ft/hr.",
        [ev(f1d12, "Rate of penetration averaged 11.7 ft/hr")])
    add("similar_wells", "Was anyone injured on Condor-7?",
        "Yes, on Day 5 a crew member sustained a minor hand laceration while handling casing tongs; treated on site, no further medical attention required.",
        [ev(c7d5, "minor hand laceration")])
    # Neither Falcon-1 report says "no injuries": Day 12 says "No lost-time incidents" and
    # Day 13 says "No incidents or near-misses reported". Answering needs a small inference.
    add("inference", "Was anyone injured on Falcon-1?",
        "The reports do not state injuries either way, and never use the words \"no injuries\". Day 12 records one near-miss (a pipe wrench left on the rig floor walkway) and no lost-time incidents; Day 13 records no incidents or near-misses. So no injury is recorded.",
        [ev(f1d12, "No lost-time incidents."), ev(f1d13, "No incidents or near-misses reported.")],
        watch=True)
    add("single_fact", "What was Falcon-1's mud weight on the day it reached casing point?",
        "11.1 ppg, on Day 13, increased from the prior day to manage anticipated pore pressure at the casing point.",
        [ev(f1d13, "Mud weight: 11.1 ppg")])
    add("single_fact", "Which well experienced downtime due to a cementing unit delay?",
        "Condor-7, on Day 5 -- 3.5 hours of NPT waiting on the cement unit, which arrived late due to a mechanical breakdown en route.",
        [ev(c7d5, "waiting on cement unit")])
    add("single_fact", "What safety incident occurred on Falcon-1 Day 12?",
        "A near-miss: a pipe wrench was left on the rig floor walkway during a connection. It was removed immediately and a toolbox talk was held with the floor crew on housekeeping standards. No lost-time incidents.",
        [ev(f1d12, "One near-miss reported")])
    add("single_fact", "What was Falcon-1's mud weight on Day 13, and why was it increased?",
        "11.1 ppg, increased from the prior day to manage anticipated pore pressure at the casing point.",
        [ev(f1d13, "Mud weight: 11.1 ppg, increased")])
    first = qs[-1]
    add("repeat", first["user_input"], first["reference"], first["evidence"], repeat_of=first["id"])
    add("not_documented", "How does the on_click method handle rapid repeated clicks in Widget Framework?",
        "The documentation does not describe any special handling for rapid repeated clicks. It only shows on_click incrementing a counter and calling self.rerender(); a debounce_ms setting exists for widget configuration, but no click-specific debouncing behavior is documented.")

    # --- new: single facts ------------------------------------------------
    h = R["Harrier-2", 4]
    add("single_fact", "What depth did Harrier-2 reach on Day 4?", f"{fmt(h['end'])} ft.",
        [ev(h["doc"], f"to {fmt(h['end'])} ft")])
    o = R["Osprey-5", 9]
    add("single_fact", "What injury occurred on Osprey-5 on Day 9?",
        "A crew member sustained a minor pinched finger while operating the catwalk pipe handler; treated on site, no further medical attention required.",
        [ev(o["doc"], "minor pinched finger")], long_doc=True)
    m = R["Merlin-8", 5]
    add("single_fact", "How much fluid was lost when lost circulation occurred on Merlin-8?",
        "40 bbl were lost on Day 5; an LCM pill was pumped and full returns were restored.",
        [ev(m["doc"], "lost circulation of 40 bbl")], long_doc=True)

    # --- new: similar wells / values that change daily --------------------
    c = R["Condor-9", 4]
    add("similar_wells", "What was the mud weight on Condor-9 on Day 4?", f"{c['weight']:.1f} ppg.",
        [ev(c["doc"], f"Mud weight: {c['weight']:.1f} ppg.")])
    f = R["Falcon-3", 12]
    add("similar_wells", "What was the average rate of penetration on Falcon-3 Day 12?", f"{f['rop']} ft/hr.",
        [ev(f["doc"], f"Rate of penetration averaged {f['rop']} ft/hr")])
    add("similar_wells", "What was the mud weight on Falcon-3 on Day 12?", f"{f['weight']:.1f} ppg.",
        [ev(f["doc"], f"Mud weight: {f['weight']:.1f} ppg.")], long_doc=True)
    h2 = R["Harrier-2", 2]
    add("similar_wells", "What was Harrier-2's mud weight on Day 2?", f"{h2['weight']:.1f} ppg.",
        [ev(h2["doc"], f"Mud weight: {h2['weight']:.1f} ppg.")])
    c6 = R["Condor-9", 6]
    add("similar_wells", "Who prepared the Condor-9 report for Day 6?", f"{c6['man']}, Company Man.",
        [ev(c6["doc"], f"Report prepared by: {c6['man']}")])

    # --- new: multi-hop (needs 2 chunks) ----------------------------------
    h5 = R["Harrier-2", 5]
    add("multi_hop", "By how much did Harrier-2's mud weight change between Day 2 and Day 5?",
        f"From {h2['weight']:.1f} ppg on Day 2 to {h5['weight']:.1f} ppg on Day 5, a change of {h5['weight'] - h2['weight']:+.1f} ppg.",
        [ev(h2["doc"], f"Mud weight: {h2['weight']:.1f} ppg."), ev(h5["doc"], f"Mud weight: {h5['weight']:.1f} ppg.")])
    o8 = R["Osprey-5", 8]
    add("multi_hop", "How many hours of non-productive time did Osprey-5 record in total on Day 8 and Day 9?",
        "3.5 hours: 2.0 hours on Day 8 (top drive hydraulic leak) and 1.5 hours on Day 9 (mud pump liner failure).",
        [ev(o8["doc"], "2.0 hours non-productive time"), ev(o["doc"], "1.5 hours non-productive time")],
        long_doc=True)
    c5 = R["Condor-9", 5]
    higher, lower = (c, c5) if c["rop"] > c5["rop"] else (c5, c)
    add("multi_hop", "Which day had the higher rate of penetration on Condor-9, Day 4 or Day 5, and by how much?",
        f"Day {higher['day']} ({higher['rop']} ft/hr versus {lower['rop']} ft/hr on Day {lower['day']}), a difference of {higher['rop'] - lower['rop']:.1f} ft/hr.",
        [ev(c["doc"], f"averaged {c['rop']} ft/hr"), ev(c5["doc"], f"averaged {c5['rop']} ft/hr")])
    k16, k17 = R["Kestrel-4", 16], R["Kestrel-4", 17]
    add("multi_hop", "How many feet did Kestrel-4 drill in total across Day 16 and Day 17?",
        f"{fmt(k16['end'] - k16['start'] + k17['end'] - k17['start'])} ft: {fmt(k16['end'] - k16['start'])} ft on Day 16 and {fmt(k17['end'] - k17['start'])} ft on Day 17.",
        [ev(k16["doc"], f"from {fmt(k16['start'])} ft to {fmt(k16['end'])} ft"),
         ev(k17["doc"], f"from {fmt(k17['start'])} ft to {fmt(k17['end'])} ft")])

    # --- new: not documented (hallucination test) -------------------------
    add("not_documented", "What was the H2S level recorded on Osprey-5?", "Not documented in the reports.")
    add("not_documented", "Who is the toolpusher on Rig 88?", "Not documented. The reports name only the company man.")
    add("not_documented", "What was the daily cost of Harrier-2 on Day 3?", "Not documented in the reports.")
    add("not_documented", "What was the hole inclination at the end of Kestrel-4 Day 18?", "Not documented in the reports.")

    # --- new: keyword-heavy (exact IDs / numbers, BM25 should win) --------
    b = R["Merlin-8", 3]
    add("keyword", f"Which well and day used bit serial number {b['bit']}?", f"{b['well']}, Day {b['day']}.",
        [ev(b["doc"], f"bit serial number {b['bit']}")])
    w = next(r for r in R.values() if r["wo"] and r["well"] == "Kestrel-4")
    add("keyword", f"What caused the downtime recorded under work order {w['wo']}?",
        f"{w['cause'][0].upper() + w['cause'][1:]} ({w['well']}, Day {w['day']}, {w['npt_h']:.1f} hours).",
        [ev(w["doc"], f"Work order {w['wo']}")])
    add("keyword", "Which report lists chlorides of 19,200 mg/L?", "Falcon-1, Day 13.",
        [ev(f1d13, "Chlorides: 19,200 mg/L.")])
    t = R["Osprey-5", 11]
    add("keyword", f"Which well and day reached a depth of {fmt(t['end'])} ft?", f"{t['well']}, Day {t['day']}.",
        [ev(t["doc"], f"to {fmt(t['end'])} ft")])
    # --- NEW, added after the BM25 weighting rule was designed on the 4 keyword
    # --- questions above. They were NOT used to design or tune anything.
    docs = {p.name: p.read_text(encoding="utf-8") for p in DOCS_DIR.glob("ddr_*.md")}
    docs.update({r["doc"]: r["text"] for r in R.values()})

    def count(text):  # how many documents contain this exact phrase
        return sum(text in t for t in docs.values())

    def first_unique(keys, phrase_fn):
        """First (well, day) in a fixed list whose phrase appears in exactly 1 document."""
        for key in keys:
            if count(phrase_fn(R[key])) == 1:
                return R[key]
        raise AssertionError("no unique candidate")

    holdout = {"holdout": True}
    h = R["Harrier-2", 5]
    add("keyword_holdout", f"Which well and day used bit serial number {h['bit']}?", f"{h['well']}, Day {h['day']}.",
        [ev(h["doc"], f"bit serial number {h['bit']}")], **holdout)
    w = next(r for r in R.values() if r["wo"] and r["well"] == "Merlin-8")
    add("keyword_holdout", f"Which report is work order {w['wo']} in, and what was the cause?",
        f"{w['well']}, Day {w['day']}: {w['cause']} ({w['npt_h']:.1f} hours).", [ev(w["doc"], f"Work order {w['wo']}")], **holdout)
    f13 = R["Falcon-3", 13]
    add("keyword_holdout", f"On which well and day did drilling end at {fmt(f13['end'])} ft?", f"{f13['well']}, Day {f13['day']}.",
        [ev(f13["doc"], f"to {fmt(f13['end'])} ft")], **holdout)
    chl_re = re.compile(r"Chlorides: ([\d,]+) mg/L\.")
    k = first_unique([("Kestrel-4", d) for d in range(15, 21)], lambda r: chl_re.search(r["text"]).group(0))
    chl = chl_re.search(k["text"]).group(0)
    add("keyword_holdout", f"Which report lists {chl[:-1]}?", f"{k['well']}, Day {k['day']}.", [ev(k["doc"], chl)], **holdout)
    c = first_unique([("Condor-9", d) for d in range(3, 9)], lambda r: f"averaged {r['rop']} ft/hr")
    add("keyword_holdout", f"Which report states an average rate of penetration of {c['rop']} ft/hr?", f"{c['well']}, Day {c['day']}.",
        [ev(c["doc"], f"averaged {c['rop']} ft/hr")], **holdout)
    o = first_unique([("Osprey-5", d) for d in range(7, 13)], lambda r: f"Mud weight: {r['weight']:.1f} ppg.")
    add("keyword_holdout", f"Which report lists a mud weight of {o['weight']:.1f} ppg?", f"{o['well']}, Day {o['day']}.",
        [ev(o["doc"], f"Mud weight: {o['weight']:.1f} ppg.")], **holdout)

    # --- NEW harder questions: two wells or two days, close numbers ----------
    def num(doc, pattern):
        return float(re.search(pattern, docs[doc]).group(1).replace(",", ""))

    def line(doc, pattern):
        return re.search(pattern, docs[doc]).group(0)

    MUD, ROP = r"Mud weight: ([\d.]+) ppg", r"averaged ([\d.]+) ft/hr"

    def compare(question, doc_a, name_a, doc_b, name_b, pattern, unit, noun):
        a, b = num(doc_a, pattern), num(doc_b, pattern)
        assert a != b, question
        hi, lo = (name_a, name_b) if a > b else (name_b, name_a)
        add("hard", question, f"{hi} had the higher {noun}: {max(a, b):,g} {unit} versus {min(a, b):,g} {unit} for {lo}, a difference of {abs(a - b):.1f} {unit}.",
            [ev(doc_a, line(doc_a, pattern)), ev(doc_b, line(doc_b, pattern))], hard_type="two_wells")

    def doc(well, day):
        return R[well, day]["doc"]

    c7d4, c7d5 = "ddr_condor7_day04.md", "ddr_condor7_day05.md"
    compare("Which had the higher mud weight on Day 4, Condor-7 or Condor-9, and by how much?",
            c7d4, "Condor-7", doc("Condor-9", 4), "Condor-9", MUD, "ppg", "mud weight")
    compare("Which was drilling faster on Day 12, Falcon-1 or Falcon-3?",
            "ddr_falcon1_day12.md", "Falcon-1", doc("Falcon-3", 12), "Falcon-3", ROP, "ft/hr", "rate of penetration")
    compare("Which had the higher funnel viscosity on Day 12, Falcon-1 or Falcon-3?",
            "ddr_falcon1_day12.md", "Falcon-1", doc("Falcon-3", 12), "Falcon-3", r"Funnel viscosity: (\d+) sec/qt", "sec/qt", "funnel viscosity")
    compare("Which had the higher chlorides on Day 13, Falcon-1 or Falcon-3?",
            "ddr_falcon1_day13.md", "Falcon-1", doc("Falcon-3", 13), "Falcon-3", r"Chlorides: ([\d,]+) mg/L", "mg/L", "chloride level")
    compare("Which had the higher mud weight on Day 5, Condor-7 or Condor-9?",
            c7d5, "Condor-7", doc("Condor-9", 5), "Condor-9", MUD, "ppg", "mud weight")

    f12 = R["Falcon-3", 12]
    f3_npt = f12["npt_h"]
    f3_evidence = f"{f3_npt:.1f} hours non-productive time" if f3_npt else "No non-productive time recorded"
    add("hard", "What was the combined non-productive time of Falcon-1 and Falcon-3 on Day 12?",
        f"{2.5 + f3_npt:g} hours: 2.5 hours on Falcon-1 and {f3_npt:g} hours on Falcon-3.",
        [ev("ddr_falcon1_day12.md", "2.5 hours non-productive time"), ev(f12["doc"], f3_evidence)], hard_type="two_wells", long_doc=True)
    crew = lambda d: int(re.search(r"(\d+) rig crew", docs[d]).group(1))
    add("hard", "How many rig crew were on site in total on Day 12, counting Falcon-1 and Falcon-3?",
        f"{crew('ddr_falcon1_day12.md') + crew(f12['doc'])}: {crew('ddr_falcon1_day12.md')} on Falcon-1 and {crew(f12['doc'])} on Falcon-3.",
        [ev("ddr_falcon1_day12.md", f"{crew('ddr_falcon1_day12.md')} rig crew"), ev(f12["doc"], f"{crew(f12['doc'])} rig crew")],
        hard_type="two_wells", long_doc=True)

    k16, k19 = R["Kestrel-4", 16], R["Kestrel-4", 19]
    delta = k19["weight"] - k16["weight"]
    assert delta != 0
    add("hard", "By how much did Kestrel-4's mud weight change between Day 16 and Day 19?",
        f"From {k16['weight']:.1f} ppg on Day 16 to {k19['weight']:.1f} ppg on Day 19, a change of {delta:+.1f} ppg.",
        [ev(k16["doc"], f"Mud weight: {k16['weight']:.1f} ppg."), ev(k19["doc"], f"Mud weight: {k19['weight']:.1f} ppg.")],
        hard_type="two_days", long_doc=True)
    h5, h6 = R["Harrier-2", 5], R["Harrier-2", 6]
    add("hard", "How many feet did Harrier-2 drill on Day 5 and Day 6 combined?",
        f"{fmt(h6['end'] - h5['start'])} ft: {fmt(h5['end'] - h5['start'])} ft on Day 5 and {fmt(h6['end'] - h6['start'])} ft on Day 6.",
        [ev(h5["doc"], f"from {fmt(h5['start'])} ft to {fmt(h5['end'])} ft"), ev(h6["doc"], f"from {fmt(h6['start'])} ft to {fmt(h6['end'])} ft")],
        hard_type="two_days")
    return qs


def validate(questions, docs):
    """Every gold evidence string must really be in its document, and
    keyword evidence must be unique across the corpus."""
    for q in questions:
        for e in q["evidence"]:
            assert e["text"] in docs[e["doc"]], (q["id"], e)
        if q["category"] in ("keyword", "keyword_holdout"):
            hits = [d for d, t in docs.items() if q["evidence"][0]["text"] in t]
            assert len(hits) == 1, (q["id"], hits)
        if q["category"] == "not_documented":
            for term in NOT_DOCUMENTED_TERMS:
                assert not any(term in t.lower() for t in docs.values()), term
    assert len({q["id"] for q in questions}) == len(questions)
    assert sum(q["category"] == "repeat" for q in questions) == 1


def main():
    rng = random.Random(SEED)
    reports = build_reports(rng)
    for r in reports.values():
        (DOCS_DIR / r["doc"]).write_text(r["text"], encoding="utf-8")

    docs = {p.name: p.read_text(encoding="utf-8") for p in DOCS_DIR.glob("ddr_*.md")}
    questions = build_questions(reports)
    validate(questions, docs)
    EVAL_SET_PATH.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")

    cats = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"Wrote {len(reports)} reports to {DOCS_DIR.name}/ ({len(docs)} DDRs in total)")
    print(f"Wrote {len(questions)} questions to {EVAL_SET_PATH.relative_to(ROOT)}: {cats}")


if __name__ == "__main__":
    main()
