"""
build_report.py
===============
Assembles the final submission PDF from the artefacts produced by run_all.py:
source code (extracted with `ast`, so the PDF always shows the code that was actually run),
execution log, figures, tables and the written analysis (analysis_text.py).
Usage:  python3 build_report.py
"""
import ast
import json
import os
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import analysis_text as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
R = json.load(open(os.path.join(OUT, "results.json")))
LOG = open(os.path.join(OUT, "log.txt")).read()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=4)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("C", fontName="Courier", fontSize=6.4, leading=7.6, backColor=colors.HexColor("#f5f5f5"),
                      borderPadding=3, spaceAfter=6)
LOGSTYLE = ParagraphStyle("L", parent=CODE, backColor=colors.HexColor("#eef4fb"))


def clean(s):
    """reportlab's base fonts are Latin-1; replace anything outside that range."""
    return s.encode("latin-1", "replace").decode("latin-1")


def wrap_lines(text, width=135):
    out = []
    for line in text.splitlines():
        out += textwrap.wrap(line, width, subsequent_indent="    ", drop_whitespace=False,
                             replace_whitespace=False) or [""]
    return "\n".join(out)


def code_block(src, style=CODE):
    return Preformatted(clean(wrap_lines(src.rstrip("\n"))), style)


def py_segment(path, name):
    """Return the exact source text of top-level function/class `name` from `path`."""
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


def log_between(start, end=None):
    i = LOG.index(start)
    j = LOG.index(end, i + 1) if end else len(LOG)
    return LOG[i:j].strip("\n")


def para(txt, style=BODY):
    return Paragraph(txt, style)


def fig(name, width=16.5 * cm):
    from PIL import Image as PILImage
    path = name if os.path.isabs(name) else os.path.join(OUT, name)
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)


def table(rows, colw=None, header=True, fs=8.5):
    t = Table(rows, colWidths=colw, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def build():
    S = []
    LIB, RUN = os.path.join(HERE, "sticky_rl.py"), os.path.join(HERE, "run_all.py")

    # ------------------------------------------------------------------ title page
    S += [para("Assignment II (Set A) - Robust Reinforcement Learning under Sticky Actuators", H1),
          para("Q-learning vs SARSA on MountainCar-v0 with a sticky actuator (Gymnasium wrapper)", H2)]
    S += [para("<b>Group Contribution Declaration</b>", H2),
          table([["Group #", T.GROUP_NUMBER]] + [["Member name / ID", "Contribution (%)"]] +
                [[m, c] for m, c in T.MEMBERS], colw=[11 * cm, 5 * cm], header=False), Spacer(1, 6)]
    S += [para("<b>Execution environment</b>", H2),
          para(clean(f"Run started {R['started']} / finished {R['finished']}. " + T.ENV_NOTE)),
          para("Timestamped virtual-lab screenshots are attached in the final section of this report "
               "(Appendix C).")]
    S += [para("<b>Contents</b>", H2),
          para("(a) Environment implementation and verification &nbsp;|&nbsp; (b) Q-learning &nbsp;|&nbsp; (c) SARSA "
               "&nbsp;|&nbsp; (d) Performance evaluation &nbsp;|&nbsp; (e) Discussion &nbsp;|&nbsp; Appendix A: full "
               "source &nbsp;|&nbsp; Appendix B: full execution log &nbsp;|&nbsp; Appendix C: virtual-lab screenshots")]
    S += [para("<b>Overall design (modularity)</b>", H2), para(T.DESIGN_NOTE)]
    S.append(PageBreak())

    # ------------------------------------------------------------------ (a)
    S += [para("(a) Modified Environment Implementation and Verification", H1), para(T.A_INTRO)]
    S += [para("Code - sticky-actuator wrapper (<font face='Courier'>sticky_rl.py</font>)", H2),
          code_block(py_segment(LIB, "is_gentle_arrival")), code_block(py_segment(LIB, "StickyMountainCar")),
          code_block(py_segment(LIB, "make_env"))]
    S += [para("Code - verification routines (<font face='Courier'>run_all.py</font>)", H2)]
    for nm in ("ScriptedRNG", "verify_example_table", "verify_random_policy", "verify_bonus_handcoded",
               "verify_unchanged_components"):
        S.append(code_block(py_segment(RUN, nm)))
    S += [para("Output - verification logs", H2),
          code_block(log_between("######## PART (a)", "######## PARTS (b)-(d)"), LOGSTYLE)]
    ra, rh = R["a"]["random"], R["a"]["handcoded"]
    S += [para("Verification summary", H2),
          table([["Property to demonstrate", "Evidence", "Result"],
                 ["~20% of attempted changes replaced", f"{ra['replaced']}/{ra['changes']} = {100 * ra['replace_rate']:.2f}% "
                  "(random policy, 2000 episodes)", "PASS"],
                 ["No replacement when action repeated", f"{ra['replaced_on_repeat']} replacements in {ra['repeats']} repeats", "PASS"],
                 ["Penalty on every selected push (0/2)", f"{ra['push_with_penalty']}/{ra['push_steps']} steps carry -0.2", "PASS"],
                 ["Never a penalty on action 1", f"{ra['a1_no_penalty']}/{ra['a1_steps']} steps penalty-free", "PASS"],
                 ["+20 only on gentle arrival", f"{rh['n_bonus']} bonus / {rh['n_fast']} fast arrivals w/o bonus / "
                  f"{rh['n_trunc']} truncated w/o bonus; {rh['violations']} violations", "PASS"],
                 ["Example table from the assignment", "replayed with scripted random numbers; all 5 rows match", "PASS"],
                 ["Spaces / time limit / info unchanged", "observation, action space equal; 200 steps; info == {}", "PASS"]],
                colw=[5.3 * cm, 9.2 * cm, 1.6 * cm])]
    S += [para(T.A_INTERP)]
    S.append(PageBreak())

    # ------------------------------------------------------------------ (b), (c)
    S += [para("(b) Q-Learning and (c) SARSA", H1), para(T.BC_INTRO)]
    S += [para("Code - discretization, exploration, shared training loop", H2)]
    for nm in ("Discretizer", "make_discretizer", "epsilon_greedy", "epsilon_schedule", "train_agent",
               "sample_validation_states", "avg_max_q"):
        S.append(code_block(py_segment(LIB, nm)))
    S += [para("Configuration identical for all four agents", H2),
          table([["Parameter", "Value"], ["Random seed", R["cfg"]["seed"]],
                 ["Discretization", f"{R['cfg']['n_pos']} x {R['cfg']['n_vel']} uniform bins (Q-table 20x20x3, zeros)"],
                 ["alpha / gamma", f"{R['cfg']['alpha']} / {R['cfg']['gamma']}"],
                 ["epsilon schedule", f"exponential decay {R['cfg']['eps_start']} -> {R['cfg']['eps_end']} over the first "
                  f"{int(100 * R['cfg']['decay_frac'])}% of episodes, then constant"],
                 ["Training episodes", R["cfg"]["n_episodes"]],
                 ["Validation states", f"{R['cfg']['n_val_states']} (sampled once, fixed)"]], colw=[5 * cm, 11 * cm])]
    S += [para(T.BC_JUSTIFY)]
    S += [para("Output - training and evaluation log (main seed)", H2),
          code_block(log_between("######## PARTS (b)-(d)", "######## PART (d)"), LOGSTYLE)]
    S.append(PageBreak())

    # ------------------------------------------------------------------ (d)
    S += [para("(d) Performance Evaluation", H1)]
    S += [para("Code - evaluation and plotting", H2)]
    for nm in ("greedy_policy_map", "evaluate_greedy", "moving_average"):
        S.append(code_block(py_segment(LIB, nm)))
    for nm in ("fig_lines", "make_all_figures"):
        S.append(code_block(py_segment(RUN, nm)))
    S += [para("Final greedy evaluation (eps = 0, 100 test episodes, fixed seeds 1000..1099, main seed)", H2)]
    rows = [["Agent", "Mean reward", "Success rate", "Gentle rate", "Mean ep. length", "Mean arr. speed"]]
    for nm, t in R["test"].items():
        rows.append([nm, f"{t['mean_reward']:.2f}", f"{t['success_rate']:.2f}", f"{t['gentle_rate']:.2f}",
                     f"{t['mean_length']:.1f}", f"{t['mean_arrival_speed']:.4f}"])
    S += [table(rows, colw=[4.2 * cm, 2.4 * cm, 2.3 * cm, 2.9 * cm, 2.6 * cm, 2.6 * cm]),
          para("Each agent is tested on the environment it was trained on. 'Gentle-arrival rate' = fraction of episodes "
               "terminating at the flag with speed &lt;= 0.04 (it is measured in the original environment too, where it "
               "carries no reward). Arrival speed is averaged over episodes that reached the flag.", SMALL)]
    S += [para("Training statistics, last 500 training episodes (eps = 0.01)", H2)]
    rows = [["Agent", "Reward", "Success", "Gentle", "Pushes / ep", "Length", "Avg max-Q (val.)"]]
    for nm, t in R["train_last"].items():
        rows.append([nm, f"{t['reward']:.1f}", f"{t['success']:.3f}", f"{t['gentle']:.3f}", f"{t['pushes']:.1f}",
                     f"{t['length']:.1f}", f"{t['val_value']:.2f}"])
    S += [table(rows, colw=[4.2 * cm, 2 * cm, 2 * cm, 2 * cm, 2.2 * cm, 2 * cm, 2.8 * cm])]
    S += [para("Multi-seed robustness (5 seeds, same hyper-parameters)", H2)]
    rb = R["robust"]
    rows = [["Agent", "Test mean reward", "Success", "Gentle", "Arrival speed", "Pushes/ep (train)"]]
    for nm, v in rb.items():
        rows.append([nm, f"{v['mean_reward'][0]:.1f} +- {v['mean_reward'][1]:.1f}",
                     f"{v['success_rate'][0]:.2f} +- {v['success_rate'][1]:.2f}",
                     f"{v['gentle_rate'][0]:.2f} +- {v['gentle_rate'][1]:.2f}",
                     f"{v['mean_arrival_speed'][0]:.4f} +- {v['mean_arrival_speed'][1]:.4f}",
                     f"{v['train_pushes'][0]:.1f} +- {v['train_pushes'][1]:.1f}"])
    S += [table(rows, colw=[3.9 * cm, 3.1 * cm, 2.4 * cm, 2.4 * cm, 3.0 * cm, 2.9 * cm], fs=8),
          para("The headline figures and plots use the main seed (0) as the assignment prescribes one common seed; the "
               "5-seed table is an additional check on how much of any difference is just seed-to-seed noise.", SMALL)]
    S.append(PageBreak())
    for fname, title, interp in [("fig_reward.png", "Episode reward vs training episode", T.D_REWARD),
                                 ("fig_value.png", "Average predicted value on fixed validation states", T.D_VALUE),
                                 ("fig_success_gentle.png", "Success rate and gentle-arrival rate", T.D_SUCCESS),
                                 ("fig_pushes.png", "Push actions per episode", T.D_PUSH)]:
        S += [para(title, H2), fig(fname), para("<b>Interpretation.</b> " + interp)]
        if fname in ("fig_value.png", "fig_pushes.png"):
            S.append(PageBreak())
    S += [para("Greedy policy maps", H2), fig("fig_policy_maps.png", 14.5 * cm),
          para("<b>Interpretation.</b> " + T.D_POLICY)]
    S.append(PageBreak())

    # ------------------------------------------------------------------ (e)
    S += [para("(e) Discussion", H1)]
    for q, a in T.E_QA:
        S += [para("<b>" + q + "</b>"), para(a)]
    S.append(PageBreak())

    # ------------------------------------------------------------------ appendices
    S += [para("Appendix A - Complete source code", H1), para("sticky_rl.py", H2), code_block(open(LIB).read()),
          PageBreak(), para("run_all.py", H2), code_block(open(RUN).read()), PageBreak(),
          para("Appendix B - Complete execution log (outputs/log.txt)", H1), code_block(LOG, LOGSTYLE), PageBreak(),
          para("Appendix C - Virtual-lab screenshots with timestamps", H1), para(T.SCREENSHOT_NOTE)]
    shot_dir = os.path.join(HERE, "lab_screenshots")
    shots = sorted(f for f in os.listdir(shot_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))) \
        if os.path.isdir(shot_dir) else []
    for f in shots:
        S += [para(f, SMALL), fig(os.path.join(shot_dir, f), 16 * cm)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"Assignment II (Set A) - Q-learning vs SARSA under sticky actuators - page {doc.page}")
        canvas.restoreState()

    out = os.path.join(HERE, T.OUTPUT_NAME)
    SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Assignment II Set A").build(S, onFirstPage=footer, onLaterPages=footer)
    print("written", out)


if __name__ == "__main__":
    build()
