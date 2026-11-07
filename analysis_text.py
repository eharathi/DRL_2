"""
analysis_text.py - written analysis for the report (kept separate from build_report.py so that the
layout code and the prose can be edited independently).  All numbers quoted here come from
outputs/log.txt (main seed 0 unless stated "5 seeds").
"""

OUTPUT_NAME = "Group_X - Q_learning_SARSA_Set A.pdf"   # rename X to the group number before submitting

# ------------------------------------------------------------------ front matter (TO BE COMPLETED BY THE GROUP)
GROUP_NUMBER = "<<ENTER GROUP NUMBER>>"
MEMBERS = [("<<Member 1 name / ID>>", "<<xx %>>"), ("<<Member 2 name / ID>>", "<<xx %>>"),
           ("<<Member 3 name / ID>>", "<<xx %>>"), ("<<Member 4 name / ID>>", "<<xx %>>")]
ENV_NOTE = ("Python 3.13.2, NumPy 2.1.1, Gymnasium 1.3.0, Matplotlib 3.10.1; CPU only; all experiments use fixed seeds and "
            "are bit-for-bit reproducible (two independent runs gave identical results). "
            "[GROUP TO COMPLETE: state here that this run was executed in the virtual lab, with date and time.]")
SCREENSHOT_NOTE = ("Place the timestamped virtual-lab screenshots (code running + outputs) in the folder "
                   "<font face='Courier'>lab_screenshots/</font> next to this script and re-run "
                   "<font face='Courier'>python3 build_report.py</font>; they are appended here automatically.")

DESIGN_NOTE = (
    "The code is split into reusable pieces: <font face='Courier'>StickyMountainCar</font> (the wrapper), "
    "<font face='Courier'>Discretizer</font>, one <font face='Courier'>train_agent</font> loop shared by both algorithms "
    "(the <i>only</i> line that differs between Q-learning and SARSA is the TD target, which makes the comparison fair by "
    "construction), evaluation helpers, and a driver (<font face='Courier'>run_all.py</font>) that runs every task and "
    "writes all evidence. Every function carries a docstring and every operation an inline comment. Randomness is split into "
    "three independent seeded streams (actuator, agent exploration, base-environment start states), so a run is "
    "reproducible and the actuator noise does not depend on the agent's behaviour.")

# ------------------------------------------------------------------ (a)
A_INTRO = (
    "The wrapper subclasses <font face='Courier'>gym.Wrapper</font> and implements Steps 1-6 in the prescribed order: it "
    "stores the agent's action <i>a</i>; if <i>a</i> differs from <i>a_prev</i> it draws r~U[0,1) and replaces the action by "
    "<i>a_prev</i> when r &lt; 0.20 (no random number is drawn when the action repeats); executes <i>a_exec</i> in the base "
    "environment; sets <i>a_prev = a_exec</i>; and returns R = R_base - 0.2*1[a != 1] + B, where the penalty uses the "
    "<b>agent-selected</b> action and B = +20 only for terminated, not truncated, position &gt;= 0.5 and velocity &lt;= 0.04. "
    "<i>a_prev</i> is reset to 1 on every <font face='Courier'>reset()</font>. The returned <i>info</i> is the untouched base "
    "dictionary and the observation/action spaces are not modified. Diagnostic attributes (<i>last_replaced</i>, ...) exist "
    "only on the wrapper object for the verification code and are never returned to the agent.")

A_INTERP = (
    "<b>Interpretation.</b> The replay of the assignment's example table reproduces all five executed actions exactly, "
    "including the case where step 2 (a = a_prev) consumes no random number. Over 266,817 attempted changes the empirical "
    "replacement rate is 19.87%, within sampling error of 20% (binomial standard error approx. 0.08 percentage points, so "
    "19.87% is about 1.7 standard errors away). No replacement ever occurs on repeated actions, the penalty is present on all "
    "266,584 selected pushes and absent on all 133,416 selections of action 1 (even when the sticky actuator keeps pushing), "
    "and the reward identity R = -1 - 0.2*1[a != 1] + B holds on every one of the 400,000 steps. A random policy never reached "
    "the flag, so the bonus was verified with a hand-coded policy (push along the velocity, 15% random actions): the +20 "
    "appeared in 855 episodes, never in the 610 flag arrivals faster than 0.04 (speeds 0.040-0.050) and never in the 35 "
    "truncated episodes; bonus and criterion agree on every step (0 disagreements).")

# ------------------------------------------------------------------ (b) / (c)
BC_INTRO = (
    "Both agents are tabular, with a Q-table of shape (20, 20, 3) initialised to zero, an epsilon-greedy behaviour policy and "
    "the update Q(s,a) += alpha*[target - Q(s,a)] where <i>a</i> is the action <b>selected</b> by the agent (the executed action "
    "is hidden). Q-learning uses target R + gamma*max_a' Q(s',a'); SARSA uses R + gamma*Q(s',a') with a' the next "
    "epsilon-greedy action actually chosen (that same a' is then executed, so SARSA is genuinely on-policy). On true "
    "termination the target is just R (no bootstrapping); truncation at 200 steps is <i>not</i> terminal and still "
    "bootstraps. Each of the four agents (Q-learning/SARSA x original/modified) is trained with the same seed, discretization, "
    "alpha, gamma, epsilon schedule and 5,000 episodes.")

BC_JUSTIFY = (
    "<b>Design choices.</b> <i>Discretization 20 x 20:</i> the assignment's minimum; one position bin is 0.09 wide and one "
    "velocity bin 0.007 wide, which is fine enough to represent the swing-up (the maximum speed is 0.07, so a velocity "
    "bin is one tenth of the speed range) yet small enough (400 states, 1,200 Q-values) for every cell to be visited many times "
    "in 5,000 episodes. Typically, finer grids learn more slowly with a tabular method and coarser ones alias more dynamics (we did not test other grid sizes). The velocity "
    "threshold 0.04 for the bonus is not a bin edge (edges are at multiples of 0.007), so the agent cannot read the bonus "
    "condition directly from the state; this is a source of aliasing in both environments. "
    "<i>alpha = 0.1, gamma = 0.99:</i> standard values; the reward is dense (-1 per step) and gamma = 0.99 keeps the 200-step "
    "horizon relevant. <i>Epsilon schedule:</i> exponential decay from 1.0 to 0.01 over the first 80% of episodes "
    "(eps_t = 1.0*(0.01)^(t/4000)), then constant at 0.01 for the last 1,000 episodes. Because all rewards are negative, the zero "
    "initialisation is optimistic, which itself drives exploration. No hyper-parameter was tuned per algorithm or per "
    "environment: the first configuration tried was kept, to keep the comparison unbiased.")

# ------------------------------------------------------------------ (d) interpretations
D_REWARD = (
    "Rewards are 100-episode moving averages. At the start the modified agents sit near -226 versus -200 for the original ones: "
    "with epsilon = 1 about two thirds of the 200 actions are pushes, and 133 pushes * 0.2 = -26.6 of extra penalty. All curves "
    "are noisy (a tabular agent with a discretized, partially observed state keeps oscillating between policies that reach the "
    "flag and ones that do not). Rewards of the original and modified environments are on different scales and should not be "
    "compared directly. Within the original environment SARSA ends higher than Q-learning for the main seed (last-500 mean "
    "-147.4 vs -164.5); within the modified environment SARSA also ends higher (-156.9 vs -186.0), and Q-learning shows "
    "a late dip. However, the 5-seed table shows the final greedy test rewards are statistically indistinguishable "
    "(original: -156.7 vs -159.5; modified: -174.6 vs -175.4, standard deviations of 11-22), so the main-seed ordering should "
    "not be read as a general algorithm ranking.")

D_VALUE = (
    "The average of max_a Q(s,a) over 500 fixed validation states starts at 0 (zero initialisation) and falls steadily to about "
    "-28 for all four agents; it is still decreasing at episode 5,000, so the tables are not fully converged. Q-learning "
    "lies above SARSA for most of training in both environments (plausibly because the max in its target propagates the most optimistic "
    "estimate, so the initial optimism disappears more slowly), and the curves meet near the end when epsilon is small and the two targets nearly "
    "coincide. At the end the gap (Q-learning minus SARSA) is -0.44 (original) and +0.14 (modified); averaged over the last "
    "500 episodes it is -0.36 and +0.56. Averaged over the whole training run the absolute gap is 1.38 (original) vs 1.90 "
    "(modified). Across 5 seeds the last-500 gap is +0.49 +- 0.55 (original) and +1.07 +- 1.06 (modified). The modified "
    "environment therefore shows a somewhat larger gap, but the intervals overlap and one seed (+3.16) dominates, so this is "
    "suggestive only.")

D_SUCCESS = (
    "Both algorithms start reaching the flag after roughly 900 episodes (before that the 200-step limit is hit every "
    "time) and then improve, with large oscillations. The two panels show the clearest effect of the modification. In the "
    "<b>original</b> environment the gentle-arrival rate is erratic and often far below the success rate (last-500 means: "
    "Q-learning success 0.92 but gentle 0.36; SARSA 0.99 vs 0.56), because arrival speed is not rewarded and is whatever the "
    "learned swing happens to give. In the <b>modified</b> environment the gentle rate almost equals the success rate "
    "(Q-learning 0.81 vs 0.80; SARSA 0.94 vs 0.94): the +20 bonus makes the agents learn to arrive at speed &lt;= 0.04. "
    "The modified agents' success rate is somewhat lower than the original ones for Q-learning (0.81 vs 0.92) in this seed, "
    "which is expected since the hidden sticky actuator makes control less reliable, but the 5-seed means (0.87 vs 0.97 for "
    "Q-learning, 0.82 vs 0.86 for SARSA) have large standard deviations (0.05-0.22).")

D_PUSH = (
    "All agents start near 133 pushes per episode (2/3 of 200 random actions). In the original environment nothing penalises "
    "pushing and the count stays around 125-150. In the modified environment it falls to roughly 100-135, and for the main seed "
    "ends at 123 (Q-learning) and 105 (SARSA) versus 135 and 127 in the original environment. The single-seed curves are noisy, "
    "but the 5-seed means of the last 500 training episodes confirm the shift: 113.1 +- 5.8 (Q-learning) and 115.6 +- 6.5 "
    "(SARSA) in the modified environment versus 132.2 +- 1.5 and 133.1 +- 5.1 in the original, i.e. 13-15% fewer pushes. "
    "As a share of all steps this is 68-69% of steps in the modified environment (last-500 means: 122.9/177.5 and 104.9/154.6) "
    "versus 82-86% in the original (134.6/164.5 and 126.6/147.4); pushing is still the majority action because MountainCar "
    "cannot be solved without pumping energy, and a 0.2 penalty is small compared with the -1 time cost and the +20 bonus.")

D_POLICY = (
    "All four maps show the classic swing-up structure: push right (orange) when velocity is positive and left (blue) when "
    "it is negative, so the car pumps energy into its oscillation. Cells that are never visited keep their optimistic zero "
    "value and so look arbitrary (the blue columns at the far left/right and for position &gt; 0.5, which is terminal, come "
    "from ties resolved to action 0). The maps are patchy because 5,000 episodes visit the grid unevenly. The two algorithms "
    "agree on only 65% (original) and 62% (modified) of cells, which mostly reflects different noise in rarely visited cells "
    "rather than systematically different strategies. The modified maps contain more 'no push' cells (22.5% and 23.0% of "
    "cells versus 18.2% and 15.5% in the original environment), which is the footprint of the energy penalty: where pushing and "
    "coasting are almost equally good, coasting now wins.")

# ------------------------------------------------------------------ (e)
E_QA = [
    ("1. Does the sticky actuator increase the difference between the value estimates of Q-learning and SARSA?",
     "Only weakly, and not conclusively. For the main seed the mean absolute gap between the two avg-max-Q curves over "
     "training grows from 1.38 (original) to 1.90 (modified), and the last-500-episode gap goes from -0.36 to +0.56. "
     "Over 5 seeds the last-500 gap is +0.49 +- 0.55 (original) versus +1.07 +- 1.06 (modified); it is larger in the modified "
     "environment for 4 of 5 seeds, but one seed (+3.16) drives most of the average and the intervals overlap, so we cannot "
     "claim a statistically reliable increase. This direction is plausible: with stochastic action execution the max operator "
     "in Q-learning's target picks the best of several noisy estimates, which inflates it relative to the on-policy expectation "
     "used by SARSA. However both value curves end at the same level (about -28) because exploration is small (eps = 0.01) at "
     "the end of training and the tables are not yet fully converged, which limits how far the two can differ. A cleaner test "
     "would use more seeds and longer training."),
    ("2. Why is the modified environment non-Markovian with respect to the observation, and how does it affect credit assignment?",
     "The executed action depends on a_prev, which is part of the wrapper's internal state but not of the observation (position, "
     "velocity). The next-state distribution P(s' | s, a) therefore depends on the hidden variable: the same (s, a) gives a "
     "distribution that is 100% 'a' if a_prev = a, but 80% 'a' and 20% 'a_prev' otherwise. Because a_prev is a function of the "
     "agent's own recent history, the observation alone is not a sufficient statistic: the problem is a POMDP, and the tabular "
     "TD methods (which assume the Markov property) estimate values that average over the hidden a_prev under the "
     "visitation distribution of the current policy. Credit assignment suffers in three ways: (i) the reward and the next state "
     "are caused by a_exec, but the update credits the selected action a, so Q(s,a) is partly contaminated by the effect of "
     "a_prev; (ii) the energy penalty is charged to the selected action even when it was not executed, so a push that never "
     "happened is still punished (and a stuck push after selecting 1 is free); (iii) the +20 bonus arrives only at the last "
     "step but depends on the speed built up by the preceding sequence of executed actions, so the TD updates have to "
     "propagate it through noisier transitions, which slows and destabilises learning. The coarse discretization adds "
     "further aliasing (also present in the original environment). Empirically the modified-environment curves are noisier "
     "and the 5-seed test rewards worse (-175 vs -157 to -159 for the original environment, with larger spread)."),
    ("3. Do the energy penalty and the gentle-arrival bonus encourage fewer pushes or a slower, more controlled arrival?",
     "Both, with evidence for each. <i>Fewer pushes:</i> the push-count plot shows the modified agents' counts drifting "
     "down while the original agents' stay at 125-150; the 5-seed training means are 113-116 vs 132-133 pushes per episode "
     "(13-15% fewer), and the share of 'no push' cells in the policy maps rises from 18.2% / 15.5% to 22.5% / 23.0%. "
     "<i>Slower, more controlled arrival:</i> the greedy test arrival speed falls from 0.0333 / 0.0306 (Q-learning / SARSA, "
     "original) to 0.0265 / 0.0234 (modified) over 5 seeds, and the main-seed values from 0.038 / 0.030 to 0.027 / 0.020; the "
     "gentle-arrival rate is almost identical to the success rate in the modified environment (0.71 vs 0.71 and 0.90 vs "
     "0.90 on the main-seed test set) but far below it for Q-learning in the original one (0.12 vs 0.87). The bonus (+20) "
     "dominates the incentive: the energy penalty (0.2) is small, so the agents still push in about two thirds of steps (82-86% in the original environment), and "
     "reducing pushes is a side effect rather than the main behaviour change."),
    ("4. Which algorithm performs better under sticky actuators? Is this consistent with the on-policy / off-policy theory?",
     "For the main seed SARSA is clearly better in the modified environment (greedy test reward -160.3 vs -185.8, success "
     "0.90 vs 0.71, mean length 156 vs 175). But this does <b>not</b> hold up across seeds: over 5 seeds the test reward is "
     "-174.6 +- 21.1 (SARSA) vs -175.4 +- 22.4 (Q-learning), and Q-learning has a slightly higher mean success rate (0.87 vs "
     "0.82), all well inside the seed-to-seed spread (SARSA's modified-environment reward ranges from -209.6 to -154.8 across "
     "seeds). We therefore conclude that, in our setup, neither algorithm is reliably better; the main-seed gap is mostly seed "
     "variance. This is consistent with the theory: SARSA's advantage in classic examples such as Cliff Walking comes from its "
     "target including the exploratory behaviour of the policy it follows. The sticky actuator, however, is part of the "
     "environment dynamics, so <i>both</i> algorithms learn its effect from sampled transitions, and the on-policy/off-policy "
     "distinction only matters through the exploration rate, which is just 0.01 at the end of training. The only "
     "systematic differences we observed (SARSA arrives slightly more slowly, 0.0234 vs 0.0265, and with lower value "
     "estimates during training) are small relative to the noise."),
    ("5. One limitation of the experimental setup and a possible improvement.",
     "<b>Limitation:</b> a single hidden-state-blind tabular setting with a 20 x 20 grid and 5 seeds. Hidden a_prev plus state "
     "aliasing make the learning curves very noisy (success rate oscillating between roughly 0.1 and 1.0 during training; seed-to-seed "
     "standard deviation of 11-22 in test reward), so differences of the size we are interested in between the algorithms "
     "cannot be resolved with the number of runs we made, and the value tables are still changing after 5,000 episodes. "
     "<b>Improvement:</b> run many more seeds (e.g. 30) and report confidence intervals / a paired test, train longer with a "
     "decaying learning rate, and use a better function approximator (e.g. tile coding) or eligibility traces "
     "(Q(lambda)/SARSA(lambda)), which propagate the sparse +20 bonus faster and are less sensitive to the aliasing "
     "introduced by the hidden a_prev. (Augmenting the state with the executed action would restore the Markov property, "
     "but it is excluded by the assignment because the executed action must stay hidden.)"),
]
