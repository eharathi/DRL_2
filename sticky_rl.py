"""
sticky_rl.py
============
Core library for Assignment II (Set A): Robust RL under sticky actuators.

Contents
--------
1. StickyMountainCar  - gym.Wrapper implementing the sticky actuator + modified reward.
2. Discretizer        - uniform binning of (position, velocity) into a grid.
3. train_agent        - ONE training loop used by both Q-learning and SARSA; the ONLY
                        difference between them is the TD target (guarantees a fair comparison).
4. greedy_policy_map / evaluate_greedy - helpers for the performance evaluation.
"""
import numpy as np
import gymnasium as gym

# ----------------------------------------------------------------------------------------
# Constants taken directly from the assignment text
# ----------------------------------------------------------------------------------------
STICK_PROB = 0.20        # r < 0.20  -> actuator stuck
PUSH_PENALTY = 0.2       # energy penalty for selected action in {0, 2}
ARRIVAL_BONUS = 20.0     # gentle-arrival bonus B
ARRIVAL_SPEED = 0.04     # observation[1] <= 0.04 counts as "gentle"
GOAL_POS = 0.5
N_ACTIONS = 3


def is_gentle_arrival(obs, terminated, truncated):
    """Gentle-arrival criterion (Step 5): terminated, not truncated, pos>=0.5, vel<=0.04."""
    return bool(terminated and (not truncated)
                and obs[0] >= GOAL_POS and obs[1] <= ARRIVAL_SPEED)


# ----------------------------------------------------------------------------------------
# 1. Environment wrapper
# ----------------------------------------------------------------------------------------
class StickyMountainCar(gym.Wrapper):
    """MountainCar-v0 with a sticky actuator and a modified reward.

    Internal state : a_prev (action actually executed on the previous step; 1 after reset).
    step(a):
      1. store agent action a
      2. if a != a_prev, draw r~U[0,1); if r < 0.20 -> a_exec = a_prev else a_exec = a
         (if a == a_prev nothing is drawn and a_exec = a)
      3. execute a_exec in the base env
      4. R = R_base - 0.2 * 1[a != 1] + B   (a is the AGENT's action, not a_exec)
      5. B = +20 iff terminated and not truncated and pos >= 0.5 and vel <= 0.04
      6. return obs, R, terminated, truncated, info  (info is the untouched base info)
    The agent is never told whether its action was replaced.
    """

    def __init__(self, env, seed=None, stick_prob=STICK_PROB):
        super().__init__(env)
        self.stick_prob = stick_prob
        self.a_prev = 1
        # Dedicated generator for the actuator so that results are reproducible and
        # independent of the base env / agent random streams.
        self._rng = np.random.default_rng(seed)
        # Diagnostics (NOT exposed to the agent; only read by the verification code).
        self.last_a_exec = None
        self.last_replaced = False
        self.last_r = None        # random number drawn on the last step (None if none drawn)
        self.last_bonus = 0.0

    def reset(self, **kwargs):
        self.a_prev = 1                      # "Do not accelerate" at the start of every episode
        self.last_a_exec, self.last_replaced, self.last_r, self.last_bonus = None, False, None, 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        a = int(action)                                   # Step 1: store agent's action
        # Step 2: sticky actuator
        self.last_r = None
        if a == self.a_prev:
            a_exec = a                                    # repeating the previous action: never replaced
        else:
            r = self._rng.random()                        # r ~ U[0, 1)
            self.last_r = r
            a_exec = self.a_prev if r < self.stick_prob else a
        self.last_replaced = (a_exec != a)
        self.last_a_exec = a_exec
        self.a_prev = a_exec                              # a_prev <- a_exec
        # Step 3: execute in the original env
        obs, r_base, terminated, truncated, info = self.env.step(a_exec)
        # Steps 4 & 5: modified reward (penalty uses the SELECTED action a)
        bonus = ARRIVAL_BONUS if is_gentle_arrival(obs, terminated, truncated) else 0.0
        reward = r_base - PUSH_PENALTY * (1.0 if a != 1 else 0.0) + bonus
        self.last_bonus = bonus
        # Step 6: return, info untouched
        return obs, reward, terminated, truncated, info


def make_env(modified, seed=0):
    """Build the original (modified=False) or sticky/modified (modified=True) environment."""
    env = gym.make("MountainCar-v0")                      # 200-step TimeLimit is built in
    return StickyMountainCar(env, seed=seed) if modified else env


# ----------------------------------------------------------------------------------------
# 2. State discretization
# ----------------------------------------------------------------------------------------
class Discretizer:
    """Uniform grid over [low, high] per dimension; state -> (i, j) bin indices."""

    def __init__(self, low, high, bins):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = np.asarray(bins, dtype=int)
        self.width = (self.high - self.low) / self.bins

    def __call__(self, obs):
        idx = ((np.asarray(obs, dtype=np.float64) - self.low) / self.width).astype(int)
        idx = np.clip(idx, 0, self.bins - 1)              # edge values fall in the last bin
        return int(idx[0]), int(idx[1])


def make_discretizer(n_pos=20, n_vel=20):
    """MountainCar-v0 ranges: position [-1.2, 0.6], velocity [-0.07, 0.07]."""
    return Discretizer([-1.2, -0.07], [0.6, 0.07], [n_pos, n_vel])


# ----------------------------------------------------------------------------------------
# 3. Tabular TD control (Q-learning / SARSA) - shared training loop
# ----------------------------------------------------------------------------------------
def epsilon_greedy(Q, s, eps, rng):
    """With prob eps pick a uniform random action, else argmax_a Q[s, a] (random tie-break)."""
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    q = Q[s[0], s[1]]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best)) if len(best) > 1 else int(best[0])


def epsilon_schedule(ep, n_episodes, eps_start=1.0, eps_end=0.01, decay_frac=0.8):
    """Exponential (geometric) decay from eps_start to eps_end over the first
    decay_frac * n_episodes episodes, then constant at eps_end."""
    decay_eps = decay_frac * n_episodes
    if ep >= decay_eps:
        return eps_end
    return eps_start * (eps_end / eps_start) ** (ep / decay_eps)


def train_agent(algo, modified, disc, n_episodes=5000, alpha=0.1, gamma=0.99,
                eps_start=1.0, eps_end=0.01, decay_frac=0.8, seed=0, val_states=None):
    """Train a tabular agent.

    algo     : 'q_learning' (target R + gamma*max_a' Q(s',a')) or
               'sarsa'      (target R + gamma*Q(s',a'), a' = next eps-greedy action)
    modified : True -> sticky/modified env, False -> original MountainCar-v0
    The Q-table has shape (pos_bins, vel_bins, 3) and is initialised to zero.
    Bootstrapping is skipped only on TERMINATION (truncation is not terminal).
    Returns Q and a dict of per-episode training logs.
    """
    assert algo in ("q_learning", "sarsa")
    env = make_env(modified, seed=seed)
    rng = np.random.default_rng(seed)                     # agent's exploration randomness
    Q = np.zeros((*disc.bins, N_ACTIONS))
    log = {k: np.zeros(n_episodes) for k in
           ("reward", "success", "gentle", "pushes", "length", "val_value")}

    for ep in range(n_episodes):
        eps = epsilon_schedule(ep, n_episodes, eps_start, eps_end, decay_frac)
        obs, _ = env.reset(seed=seed if ep == 0 else None)  # seed env once -> reproducible stream
        s = disc(obs)
        a = epsilon_greedy(Q, s, eps, rng)
        total, pushes, success, gentle, steps = 0.0, 0, 0, 0, 0
        done = False
        while not done:
            obs2, R, terminated, truncated, _ = env.step(a)
            s2 = disc(obs2)
            pushes += int(a != 1)                          # count SELECTED push actions
            total += R
            steps += 1
            a2 = epsilon_greedy(Q, s2, eps, rng)           # a' actually chosen by the policy
            if terminated:                                 # no bootstrap on true termination
                target = R
            elif algo == "q_learning":
                target = R + gamma * Q[s2[0], s2[1]].max()           # off-policy: max_a' Q(s',a')
            else:
                target = R + gamma * Q[s2[0], s2[1], a2]             # on-policy : Q(s',a')
            Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])  # 'a' = agent-selected action
            if terminated:
                success = 1
                gentle = int(is_gentle_arrival(obs2, terminated, truncated))
            s, a = s2, a2
            done = terminated or truncated
        log["reward"][ep], log["pushes"][ep], log["length"][ep] = total, pushes, steps
        log["success"][ep], log["gentle"][ep] = success, gentle
        if val_states is not None:                         # avg_s max_a Q(s,a) on fixed states
            log["val_value"][ep] = avg_max_q(Q, disc, val_states)
    env.close()
    return Q, log


def sample_validation_states(n=500, seed=12345):
    """Fixed validation set sampled ONCE (uniformly over the state box) before training."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.2, 0.6, n)
    vel = rng.uniform(-0.07, 0.07, n)
    return np.stack([pos, vel], axis=1)


def avg_max_q(Q, disc, states):
    """Average predicted value: mean over validation states of max_a Q(s, a)."""
    return float(np.mean([Q[disc(s)].max() for s in states]))


# ----------------------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------------------
def greedy_policy_map(Q):
    """argmax_a Q over the grid -> array (pos_bins, vel_bins) of greedy actions."""
    return Q.argmax(axis=2)


def evaluate_greedy(Q, disc, modified, n_episodes=100, base_seed=1000):
    """Run the greedy policy (eps=0) for n_episodes with fixed seeds (base_seed + i)."""
    env = make_env(modified, seed=base_seed)
    rewards, succ, gent, lens, speeds = [], [], [], [], []
    for i in range(n_episodes):
        if modified:
            env._rng = np.random.default_rng(base_seed + i)   # fixed sticky stream per test episode
        obs, _ = env.reset(seed=base_seed + i)
        total, steps, done = 0.0, 0, False
        while not done:
            q = Q[disc(obs)]
            a = int(np.flatnonzero(q == q.max())[0])           # deterministic greedy
            obs, R, term, trunc, _ = env.step(a)
            total += R
            steps += 1
            done = term or trunc
        rewards.append(total)
        succ.append(int(term))
        gent.append(int(is_gentle_arrival(obs, term, trunc)))
        lens.append(steps)
        if term:
            speeds.append(obs[1])
    env.close()
    return {"mean_reward": float(np.mean(rewards)), "success_rate": float(np.mean(succ)),
            "gentle_rate": float(np.mean(gent)), "mean_length": float(np.mean(lens)),
            "mean_arrival_speed": float(np.mean(speeds)) if speeds else float("nan")}


def moving_average(x, w=100):
    """Trailing moving average over the previous w episodes (shorter window at the start)."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out
