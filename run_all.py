"""
run_all.py
==========
Runs every task of Assignment II (Set A) and writes all evidence to ./outputs:
  outputs/log.txt        - everything printed (verification logs, tables)
  outputs/results.json   - numeric results (used by build_report.py)
  outputs/fig_*.png      - all plots
Usage:  python3 run_all.py
"""
import json
import os
import sys
import time
import datetime
import platform

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sticky_rl import (StickyMountainCar, make_env, make_discretizer, train_agent,
                       sample_validation_states, evaluate_greedy, greedy_policy_map,
                       moving_average, is_gentle_arrival, STICK_PROB, PUSH_PENALTY,
                       ARRIVAL_BONUS, ARRIVAL_SPEED)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")


def P(*args):
    """print() to both the console and outputs/log.txt."""
    msg = " ".join(str(a) for a in args)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


# ---------------------------------------------------------------------------------------
# Experimental configuration (IDENTICAL for Q-learning and SARSA, original and modified)
# ---------------------------------------------------------------------------------------
CFG = dict(seed=0, n_pos=20, n_vel=20, n_episodes=5000, alpha=0.1, gamma=0.99,
           eps_start=1.0, eps_end=0.01, decay_frac=0.8, n_val_states=500,
           n_test_episodes=100, test_base_seed=1000)
ROBUST_SEEDS = [1, 2, 3, 4]          # extra seeds (main seed is CFG['seed']) for robustness table
AGENTS = [("q_learning", False), ("sarsa", False), ("q_learning", True), ("sarsa", True)]


def name_of(algo, modified):
    return f"{'Q-learning' if algo == 'q_learning' else 'SARSA'} - {'Modified' if modified else 'Original'}"


STYLE = {  # colour = algorithm, line style = environment
    "Q-learning - Original": dict(color="#1f77b4", ls="--"),
    "SARSA - Original": dict(color="#d95f02", ls="--"),
    "Q-learning - Modified": dict(color="#1f77b4", ls="-"),
    "SARSA - Modified": dict(color="#d95f02", ls="-"),
}


# =======================================================================================
# PART (a): verification of the modified environment
# =======================================================================================
class ScriptedRNG:
    """Stand-in RNG that returns pre-scripted numbers (used to replay the assignment's example)."""

    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


def verify_example_table():
    """Replays the 5-step example table from the assignment with scripted random numbers."""
    P("\n[a.1] Deterministic replay of the example table in the assignment")
    env = StickyMountainCar(gym.make("MountainCar-v0"))
    env.reset(seed=0)
    agent_actions = [2, 2, 0, 0, 1]
    scripted_r = [0.55, 0.11, 0.74, 0.05]            # step 2 (a == a_prev) draws NO random number
    expected = [2, 2, 2, 0, 0]
    env._rng = ScriptedRNG(scripted_r)
    P(f"{'Step':<5}{'a_prev':<8}{'a':<4}{'r':<8}{'a_exec':<8}{'expected':<9}{'ok'}")
    all_ok = True
    for i, a in enumerate(agent_actions):
        a_prev = env.a_prev
        env.step(a)
        ok = env.last_a_exec == expected[i]
        all_ok &= ok
        r_txt = "-" if env.last_r is None else f"{env.last_r:.2f}"
        P(f"{i + 1:<5}{a_prev:<8}{a:<4}{r_txt:<8}{env.last_a_exec:<8}{expected[i]:<9}{ok}")
    assert not env._rng.values, "step 2 must not consume a random number"
    P("Example table reproduced exactly:", all_ok)
    return bool(all_ok)


def verify_random_policy(n_episodes=2000, seed=7):
    """Random policy: replacement rate, no-replacement-on-repeat, penalty and bonus checks."""
    P(f"\n[a.2] Random policy, {n_episodes} episodes")
    env = StickyMountainCar(gym.make("MountainCar-v0"), seed=seed)
    rng = np.random.default_rng(seed)
    n_steps = n_changes = n_replaced_on_change = n_replaced_on_repeat = n_repeats = 0
    pen_push = pen_push_ok = nopen_a1 = nopen_a1_ok = 0
    bonus_steps = bonus_wrong = reset_ok = 0
    reward_mismatch = 0
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        reset_ok += int(env.a_prev == 1)             # a_prev must be 1 after reset
        done = False
        while not done:
            a = int(rng.integers(3))
            a_prev = env.a_prev
            obs, R, term, trunc, info = env.step(a)
            done = term or trunc
            n_steps += 1
            if a == a_prev:
                n_repeats += 1
                n_replaced_on_repeat += int(env.last_replaced)
            else:
                n_changes += 1
                n_replaced_on_change += int(env.last_replaced)
            # reward identity: R == -1 - 0.2*1[a!=1] + B
            B = ARRIVAL_BONUS if is_gentle_arrival(obs, term, trunc) else 0.0
            expected = -1.0 - PUSH_PENALTY * (a != 1) + B
            reward_mismatch += int(abs(R - expected) > 1e-9)
            if a != 1:                               # push selected -> penalty must be present
                pen_push += 1
                pen_push_ok += int(abs((R - B) - (-1.0 - PUSH_PENALTY)) < 1e-9)
            else:                                    # action 1 selected -> NO penalty ever
                nopen_a1 += 1
                nopen_a1_ok += int(abs((R - B) - (-1.0)) < 1e-9)
            bonus_steps += int(B > 0)
            bonus_wrong += int((B > 0) != (env.last_bonus > 0))
            assert info == {}, "info dict must stay untouched"
    P(f"total steps                              : {n_steps}")
    P(f"attempted changes (a != a_prev)          : {n_changes}")
    P(f"  replaced by a_prev                     : {n_replaced_on_change} "
      f"({100 * n_replaced_on_change / n_changes:.2f}%  | expected {100 * STICK_PROB:.0f}%)")
    P(f"repeats (a == a_prev)                    : {n_repeats}")
    P(f"  replacements among repeats             : {n_replaced_on_repeat}  (must be 0)")
    P(f"steps with push selected (0 or 2)        : {pen_push}; penalty -0.2 present in {pen_push_ok} (must be all)")
    P(f"steps with action 1 selected             : {nopen_a1}; penalty-free in {nopen_a1_ok} (must be all)")
    P(f"reward == -1 - 0.2*1[a!=1] + B mismatches: {reward_mismatch}  (must be 0)")
    P(f"a_prev == 1 after reset                  : {reset_ok}/{n_episodes}")
    P(f"bonus steps (random policy)              : {bonus_steps}")
    return dict(steps=n_steps, changes=n_changes, replaced=n_replaced_on_change,
                replace_rate=n_replaced_on_change / n_changes, repeats=n_repeats,
                replaced_on_repeat=n_replaced_on_repeat, push_steps=pen_push,
                push_with_penalty=pen_push_ok, a1_steps=nopen_a1, a1_no_penalty=nopen_a1_ok,
                reward_mismatch=reward_mismatch, reset_ok=reset_ok, n_episodes=n_episodes)


def verify_bonus_handcoded(n_episodes=1500, seed=11, noise=0.15):
    """Hand-coded policy (push in direction of velocity, with some noise) to reach the flag often.
    Checks that +20 is awarded iff terminated & not truncated & pos>=0.5 & vel<=0.04."""
    P(f"\n[a.3] Hand-coded policy (push along velocity, {int(noise * 100)}% random actions), {n_episodes} episodes")
    env = StickyMountainCar(gym.make("MountainCar-v0"), seed=seed)
    rng = np.random.default_rng(seed)
    n_term = n_trunc = n_bonus = n_fast = viol = 0
    speeds_bonus, speeds_nobonus = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if rng.random() < noise:
                a = int(rng.integers(3))
            else:
                a = 2 if obs[1] >= 0 else 0
            obs, R, term, trunc, _ = env.step(a)
            done = term or trunc
            bonus_given = env.last_bonus > 0
            criterion = bool(term and not trunc and obs[0] >= 0.5 and obs[1] <= 0.04)
            viol += int(bonus_given != criterion)    # bonus <=> criterion on EVERY step
            if bonus_given and not done:
                viol += 1                            # bonus can only appear on the last step
        if term:
            n_term += 1
            if env.last_bonus > 0:
                n_bonus += 1
                speeds_bonus.append(obs[1])
            else:
                n_fast += 1
                speeds_nobonus.append(obs[1])
        else:
            n_trunc += 1
            assert env.last_bonus == 0
    P(f"episodes reaching the flag (terminated)  : {n_term}")
    P(f"  with +20 bonus (vel <= 0.04)           : {n_bonus}  arrival speeds in [{min(speeds_bonus):.4f}, {max(speeds_bonus):.4f}]")
    P(f"  WITHOUT bonus (vel > 0.04, too fast)   : {n_fast}  arrival speeds in "
      f"[{(min(speeds_nobonus) if speeds_nobonus else float('nan')):.4f}, {(max(speeds_nobonus) if speeds_nobonus else float('nan')):.4f}]")
    P(f"truncated episodes (never any bonus)     : {n_trunc}")
    P(f"bonus/criterion disagreements            : {viol}  (must be 0)")
    return dict(n_term=n_term, n_bonus=n_bonus, n_fast=n_fast, n_trunc=n_trunc, violations=viol,
                n_episodes=n_episodes)


def verify_unchanged_components():
    """Observation / action spaces, time limit and info dict must equal the original env."""
    P("\n[a.4] Components that must remain unchanged")
    base, wrapped = gym.make("MountainCar-v0"), StickyMountainCar(gym.make("MountainCar-v0"))
    ok_obs = base.observation_space == wrapped.observation_space
    ok_act = base.action_space == wrapped.action_space
    P(f"observation space identical: {ok_obs}  ({wrapped.observation_space})")
    P(f"action space identical     : {ok_act}  ({wrapped.action_space})")
    P(f"max episode steps          : base={base.spec.max_episode_steps}, wrapped={wrapped.spec.max_episode_steps}")
    # Sanity: with a == a_prev always (always action 1) the wrapped env equals the base env exactly.
    b_obs, _ = base.reset(seed=3)
    w_obs, _ = wrapped.reset(seed=3)
    same = np.allclose(b_obs, w_obs)
    for _ in range(200):
        b = base.step(1)
        w = wrapped.step(1)
        same &= np.allclose(b[0], w[0]) and b[2] == w[2] and b[3] == w[3]
        if b[2] or b[3]:
            break
    P(f"identical trajectories when the agent only repeats a_prev=1: {bool(same)}")
    return bool(ok_obs and ok_act and same)


# =======================================================================================
# PARTS (b), (c), (d): training and evaluation
# =======================================================================================
def run_experiments(seed, val_states, disc, keep_all=True):
    """Train the four agents for one seed. Returns {name: (Q, log, test_metrics)}."""
    res = {}
    for algo, modified in AGENTS:
        nm = name_of(algo, modified)
        t0 = time.time()
        Q, log = train_agent(algo, modified, disc, n_episodes=CFG["n_episodes"], alpha=CFG["alpha"],
                             gamma=CFG["gamma"], eps_start=CFG["eps_start"], eps_end=CFG["eps_end"],
                             decay_frac=CFG["decay_frac"], seed=seed, val_states=val_states)
        test = evaluate_greedy(Q, disc, modified, CFG["n_test_episodes"], CFG["test_base_seed"])
        P(f"  seed {seed}  {nm:<24} trained in {time.time() - t0:5.1f}s | test: " +
          ", ".join(f"{k}={v:.3f}" for k, v in test.items()))
        res[nm] = (Q, log, test)
    return res


def fig_lines(res, key, title, ylabel, fname, ma=True, w=100, extra_note=None):
    """Generic 'metric vs training episode' plot for the four agents."""
    plt.figure(figsize=(9, 4.8))
    for nm, (_, log, _) in res.items():
        y = moving_average(log[key], w) if ma else log[key]
        plt.plot(np.arange(1, len(y) + 1), y, label=nm, lw=1.6, **STYLE[nm])
    plt.xlabel("Training episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()


def make_all_figures(res):
    w = CFG["n_episodes"]
    fig_lines(res, "reward", "Episode reward vs training episode (100-episode moving average)",
              "Episode reward (MA-100)", "fig_reward.png")
    fig_lines(res, "val_value", "Average predicted value  mean_s max_a Q(s,a)  on a fixed validation set",
              "Average max-Q over validation states", "fig_value.png", ma=False)
    # success rate and gentle-arrival rate: two panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, key, ttl in zip(axes, ("success", "gentle"),
                            ("Success rate (reached flag), MA-100", "Gentle-arrival rate (flag & speed <= 0.04), MA-100")):
        for nm, (_, log, _) in res.items():
            ax.plot(np.arange(1, w + 1), moving_average(log[key], 100), label=nm, lw=1.6, **STYLE[nm])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_success_gentle.png"), dpi=150)
    plt.close()
    fig_lines(res, "pushes", "Average number of push actions (0 or 2) selected per episode (MA-100)",
              "Push actions per episode (MA-100)", "fig_pushes.png")
    # greedy policy maps (2x2): rows = algorithm, columns = environment
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cmap = matplotlib.colors.ListedColormap(["#3b6fb6", "#e8e8e8", "#d95f02"])  # left / none / right
    for ax, (algo, modified) in zip(axes.ravel(), [("q_learning", False), ("q_learning", True),
                                                   ("sarsa", False), ("sarsa", True)]):
        nm = name_of(algo, modified)
        pol = greedy_policy_map(res[nm][0])
        # rows of imshow = velocity (origin lower), columns = position
        ax.imshow(pol.T, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto",
                  extent=[-1.2, 0.6, -0.07, 0.07], interpolation="nearest")
        ax.set_title(nm)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.axvline(0.5, color="k", lw=0.8, ls=":")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#3b6fb6", "#e8e8e8", "#d95f02")]
    fig.legend(handles, ["0: push left", "1: no push", "2: push right"], loc="lower center", ncol=3)
    fig.suptitle("Greedy policy maps (argmax_a Q) over the position x velocity grid")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(OUT, "fig_policy_maps.png"), dpi=150)
    plt.close()


def last_mean(log, key, n=500):
    return float(np.mean(log[key][-n:]))


def main():
    t_start = datetime.datetime.now()
    P("=" * 90)
    P("Assignment II (Set A) - Robust RL under Sticky Actuators: execution log")
    P(f"Run started : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    P(f"Platform    : {platform.platform()} | Python {platform.python_version()} | "
      f"numpy {np.__version__} | gymnasium {gym.__version__}")
    P(f"Config      : {CFG}")
    P("=" * 90)

    # ---------------- Part (a) ----------------
    P("\n######## PART (a): ENVIRONMENT VERIFICATION ########")
    a_results = dict(example_ok=verify_example_table(), random=verify_random_policy(),
                     handcoded=verify_bonus_handcoded(), unchanged_ok=verify_unchanged_components())

    # ---------------- Parts (b)-(d) ----------------
    P("\n######## PARTS (b)-(d): TRAINING ########")
    disc = make_discretizer(CFG["n_pos"], CFG["n_vel"])
    val_states = sample_validation_states(CFG["n_val_states"], seed=12345)   # sampled ONCE, fixed
    P(f"Discretization: {CFG['n_pos']} x {CFG['n_vel']} uniform bins; Q-table shape "
      f"{(CFG['n_pos'], CFG['n_vel'], 3)}; validation states: {len(val_states)} (fixed)")
    P(f"Main run (seed {CFG['seed']}):")
    res = run_experiments(CFG["seed"], val_states, disc)
    make_all_figures(res)

    # ---------------- Summary tables ----------------
    P("\n######## PART (d): FINAL GREEDY EVALUATION (eps = 0, 100 test episodes, fixed seeds) ########")
    P(f"{'Agent':<26}{'Mean reward':>12}{'Success':>9}{'Gentle':>8}{'Mean len':>10}{'Arr. speed':>12}")
    for nm, (_, _, t) in res.items():
        P(f"{nm:<26}{t['mean_reward']:>12.2f}{t['success_rate']:>9.2f}{t['gentle_rate']:>8.2f}"
          f"{t['mean_length']:>10.1f}{t['mean_arrival_speed']:>12.4f}")

    P("\nTraining statistics over the LAST 500 training episodes (eps = 0.01):")
    P(f"{'Agent':<26}{'Reward':>9}{'Success':>9}{'Gentle':>8}{'Pushes/ep':>11}{'Length':>8}{'Avg max-Q':>11}")
    train_last = {}
    for nm, (_, log, _) in res.items():
        row = {k: last_mean(log, k) for k in ("reward", "success", "gentle", "pushes", "length")}
        row["val_value"] = float(log["val_value"][-1])
        train_last[nm] = row
        P(f"{nm:<26}{row['reward']:>9.1f}{row['success']:>9.3f}{row['gentle']:>8.3f}"
          f"{row['pushes']:>11.1f}{row['length']:>8.1f}{row['val_value']:>11.2f}")

    # value-estimate gap Q-learning minus SARSA (on the fixed validation states)
    P("\nValue-estimate gap  (Q-learning - SARSA) of avg max-Q on validation states:")
    gaps = {}
    for env_nm in ("Original", "Modified"):
        q = res[f"Q-learning - {env_nm}"][1]["val_value"]
        s = res[f"SARSA - {env_nm}"][1]["val_value"]
        gaps[env_nm] = dict(final=float(q[-1] - s[-1]), last500=float(np.mean(q[-500:] - s[-500:])),
                            mean_abs=float(np.mean(np.abs(q - s))),
                            q_final=float(q[-1]), s_final=float(s[-1]))
        P(f"  {env_nm:<9}: final gap = {gaps[env_nm]['final']:+.3f}   last-500 mean gap = "
          f"{gaps[env_nm]['last500']:+.3f}   mean |gap| over training = {gaps[env_nm]['mean_abs']:.3f}")

    # greedy policy agreement between algorithms (fraction of visited-or-not grid cells with same action)
    P("\nGreedy-policy agreement (fraction of grid cells with identical greedy action):")
    agree = {}
    for env_nm in ("Original", "Modified"):
        pa = greedy_policy_map(res[f"Q-learning - {env_nm}"][0])
        pb = greedy_policy_map(res[f"SARSA - {env_nm}"][0])
        agree[env_nm] = float(np.mean(pa == pb))
        P(f"  Q-learning vs SARSA, {env_nm}: {agree[env_nm]:.3f}")

    # push-action share on the policy map
    P("\nShare of grid cells whose greedy action is: left / none / right")
    share = {}
    for nm, (Q, _, _) in res.items():
        pol = greedy_policy_map(Q)
        share[nm] = [float(np.mean(pol == k)) for k in range(3)]
        P(f"  {nm:<24}: {share[nm][0]:.3f} / {share[nm][1]:.3f} / {share[nm][2]:.3f}")

    # ---------------- Robustness over seeds ----------------
    P("\n######## ROBUSTNESS: additional seeds (same hyper-parameters) ########")
    all_seeds = {CFG["seed"]: {nm: dict(test=t, train_last=train_last[nm]) for nm, (_, _, t) in res.items()}}
    gap_seeds = {CFG["seed"]: {e: gaps[e]["last500"] for e in gaps}}
    for sd in ROBUST_SEEDS:
        r = run_experiments(sd, val_states, disc)
        all_seeds[sd] = {nm: dict(test=t, train_last={k: last_mean(l, k) for k in ("reward", "success", "gentle", "pushes")})
                         for nm, (_, l, t) in r.items()}
        gap_seeds[sd] = {e: float(np.mean(r[f"Q-learning - {e}"][1]["val_value"][-500:]
                                           - r[f"SARSA - {e}"][1]["val_value"][-500:])) for e in ("Original", "Modified")}
    P("\nGreedy test metrics, mean +- std over %d seeds:" % len(all_seeds))
    cols = ("mean_reward", "success_rate", "gentle_rate", "mean_length", "mean_arrival_speed")
    P(f"{'Agent':<26}{'Mean reward':>16}{'Success':>13}{'Gentle':>13}{'Mean len':>15}{'Arr. speed':>16}{'Pushes/ep*':>16}")
    robust = {}
    for nm in res:
        arr = {k: np.array([all_seeds[s][nm]["test"][k] for s in all_seeds]) for k in cols}
        arr["train_pushes"] = np.array([all_seeds[s][nm]["train_last"]["pushes"] for s in all_seeds])
        robust[nm] = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in arr.items()}
        P(f"{nm:<26}" + "".join(f"{robust[nm][k][0]:>{w1}.{d}f}+-{robust[nm][k][1]:<{w2}.{d}f}"
                                for k, w1, w2, d in (("mean_reward", 9, 6, 2), ("success_rate", 7, 5, 2),
                                                     ("gentle_rate", 7, 5, 2), ("mean_length", 9, 6, 1),
                                                     ("mean_arrival_speed", 9, 6, 4), ("train_pushes", 9, 6, 1))))
    P("(* Pushes/ep = selected push actions per episode, mean of the last 500 TRAINING episodes)")
    P("\nValue gap (Q-learning - SARSA), last-500-episode mean, per seed:")
    for e in ("Original", "Modified"):
        v = [gap_seeds[s][e] for s in gap_seeds]
        P(f"  {e:<9}: " + ", ".join(f"{x:+.2f}" for x in v) + f"   -> mean {np.mean(v):+.2f} +- {np.std(v):.2f}")
    gap_robust = {e: [float(np.mean([gap_seeds[s][e] for s in gap_seeds])),
                      float(np.std([gap_seeds[s][e] for s in gap_seeds]))] for e in ("Original", "Modified")}

    t_end = datetime.datetime.now()
    P(f"\nRun finished: {t_end.strftime('%Y-%m-%d %H:%M:%S')}  (elapsed {(t_end - t_start).total_seconds():.0f}s)")

    json.dump(dict(cfg=CFG, a=a_results,
                   test={nm: t for nm, (_, _, t) in res.items()}, train_last=train_last, gaps=gaps,
                   agree=agree, share=share, robust=robust, gap_robust=gap_robust,
                   n_robust_seeds=len(all_seeds), started=t_start.isoformat(), finished=t_end.isoformat()),
              open(os.path.join(OUT, "results.json"), "w"), indent=2)
    LOG.close()


if __name__ == "__main__":
    main()
