"""make_notebook.py - builds Assignment2_Q_learning_SARSA_SetA.ipynb from the tested source files."""
import ast, re, textwrap
import nbformat as nbf

def seg(path, names):
    src = open(path).read(); out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            out.append(ast.get_source_segment(src, n))
    return "\n\n\n".join(out)

LIB, RUN = "sticky_rl.py", "run_all.py"
lib, run = open(LIB).read(), open(RUN).read()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(t).strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md("""# Assignment II (Set A): Robust RL under Sticky Actuators
Q-learning vs SARSA on `MountainCar-v0` with a sticky-actuator wrapper.

**Group #:** <<ENTER>> &nbsp; **Members / contribution %:** <<ENTER>>

*Run this notebook top to bottom (about 10 minutes). Add your virtual-lab screenshots at the end.*""")

md("## 0. Setup and shared library\nImports, constants, the gentle-arrival test, wrapper, discretizer, shared training loop and evaluation helpers. The training loop is shared by both algorithms; only the TD target differs.")
header = lib[lib.index("import numpy"):lib.index("# ----------------------------------------------------------------------------------------\n# 1. Environment wrapper")]
code(header)

md("## (a) Modified environment\n`StickyMountainCar` follows Steps 1-6 exactly: store `a`, sticky replacement with p=0.20 only when `a != a_prev`, execute `a_exec`, `R = R_base - 0.2*1[a!=1] + B`, bonus only on gentle arrival, `info` untouched, `a_prev=1` on reset.")
code(seg(LIB, ["StickyMountainCar", "make_env"]).replace("class StickyMountainCar", "def is_gentle_arrival_placeholder(): pass\n\n\nclass StickyMountainCar", 0))
cells[-1].source = seg(LIB, ["is_gentle_arrival", "StickyMountainCar", "make_env"])

md("## Discretization, exploration and the shared TD-control loop (parts b, c)\nQ-table shape (20, 20, 3) initialised to zero. Q-learning target: `R + gamma*max Q(s',.)`; SARSA target: `R + gamma*Q(s',a')`. No bootstrap on termination; truncation still bootstraps.")
code(seg(LIB, ["Discretizer", "make_discretizer", "epsilon_greedy", "epsilon_schedule", "train_agent",
               "sample_validation_states", "avg_max_q"]))
md("## Evaluation helpers")
code(seg(LIB, ["greedy_policy_map", "evaluate_greedy", "moving_average"]))

md("## Experiment configuration (identical for all four agents)")
cfg = run[run.index("OUT = os.path"):run.index("# =======================================================================================\n# PART (a)")]
code("import json, os, sys, time, datetime, platform\nimport numpy as np\nimport gymnasium as gym\nimport matplotlib\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image as ShowImage, display\n\n" + cfg.replace("os.path.dirname(os.path.abspath(__file__))", "os.getcwd()"))

md("## (a) Verification routines and results")
code(seg(RUN, ["ScriptedRNG", "verify_example_table", "verify_random_policy", "verify_bonus_handcoded", "verify_unchanged_components"]))

body = run[run.index("def main():"):]
body = textwrap.dedent("\n".join(body.splitlines()[1:]))
markers = ["# ---------------- Part (a) ----------------", "# ---------------- Parts (b)-(d) ----------------",
           "# ---------------- Summary tables ----------------", "# ---------------- Robustness over seeds ----------------"]
parts = re.split("|".join(re.escape(m) for m in markers), body.split('\nif __name__')[0])
head, pa, pbcd, summ, robust = [p.strip("\n") for p in parts]

md("## Run log header")
code(head)
code(pa)

md("## Training and plotting functions")
code(seg(RUN, ["run_experiments", "fig_lines", "make_all_figures", "last_mean"]))
md("## (b), (c) Train Q-learning and SARSA on the original and modified environments (main seed)")
code(pbcd)
md("## (d) Plots")
for f, t in [("fig_reward.png", "Episode reward (MA-100)"), ("fig_value.png", "Average max-Q on fixed validation states"),
             ("fig_success_gentle.png", "Success and gentle-arrival rate (MA-100)"),
             ("fig_pushes.png", "Push actions per episode (MA-100)"), ("fig_policy_maps.png", "Greedy policy maps")]:
    md(f"### {t}")
    code(f'display(ShowImage(filename=os.path.join(OUT, "{f}")))')
md("## (d) Final greedy evaluation and summary tables")
code(summ)
md("## Robustness over 5 seeds")
code(robust)

md(open("analysis_text.py").read() and "## (e) Discussion\nSee the cell below (rendered from the same text as the PDF report).")
import importlib, analysis_text as T
def strip(s): return re.sub(r"<font face='Courier'>(.*?)</font>", r"`\1`", s).replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*").replace("&lt;", "<").replace("&gt;", ">")
md("\n\n".join(f"**{q}**\n\n{strip(a)}" for q, a in T.E_QA))
md("## Virtual-lab screenshots\n*(attach timestamped screenshots here)*")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "drl2", "display_name": "Python 3 (DRL2)", "language": "python"}
nbf.write(nb, "Assignment2_Q_learning_SARSA_SetA.ipynb")
print("cells:", len(cells))
