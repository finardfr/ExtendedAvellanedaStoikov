import math
import np as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm
import random
import os
os.makedirs('../img', exist_ok=True)

##########################################
#       Simulations
#########################################

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*math.sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def get_alpha(t, T):
    return 0.00128 + 0.00171 * ((2*t/T - 1)**2)

n_sim = 1000

# Fixed simulation parameters
T  = 1.0
N  = 4680   # 1 step = 5s, trading day = 6.5h = 23400s
dt = T / N
t_plot = np.linspace(0.0, N*dt, N+1)

gamma         = 0.1
k             = 1.5
q_max         = 10
eta           = 0.005
kappa         = 2.0
theta         = 0.75
waiting_steps = 1
update_steps  = 1

# Baseline: fixed symmetric spread (no inventory management, no time horizon)
static_spread = 2/gamma * math.log(1 + gamma/k)

# Volatility scenarios
sigma_values = [0.5, 2.0, 5.0]
sigma_labels = ['Low vol (σ=0.5)', 'Medium vol (σ=2.0)', 'High vol (σ=5.0)']
c_opt        = 'steelblue'
c_base       = 'orange'

# Results storage per (sigma, strategy)
results = {sigma: {st: {
    'pnl_term':  np.empty(n_sim),
    'q_term':    np.empty(n_sim),
    'n_trades':  np.empty(n_sim),
    'pnl_mean':  np.zeros(N+1),
    'pnl_M2':    np.zeros(N+1),
    'q_mean':    np.zeros(N+1),
    'q_M2':      np.zeros(N+1),
} for st in ('opt', 'base')} for sigma in sigma_values}

def run_strategy(s, sigma, strategy):
    """Run one simulation day for a given price path and strategy."""
    phi_max = max(1.0, 5.0 / sigma)
    pnl = np.empty(N+2); pnl[0] = 0
    x   = np.empty(N+2); x[0]   = 0
    q   = np.empty(N+2); q[0]   = 0
    book_state = 0; time_last_event = 0
    outstanding_side = None; active_ra = 0.0; active_rb = 0.0
    trades = 0

    for n in range(N+1):
        # A-S optimal quotes
        r_n      = s[n] - q[n] * gamma * sigma**2 * (T - dt*n)
        r_spread = gamma * sigma**2 * (T - dt*n) + 2/gamma * math.log(1 + gamma/k)
        if strategy == 'opt':
            ra_n = r_n + r_spread/2
            rb_n = r_n - r_spread/2
        else:  # baseline: fixed symmetric spread, no inventory adjustment
            ra_n = s[n] + static_spread/2
            rb_n = s[n] - static_spread/2

        alpha_t = get_alpha(n*dt, T)
        dNa = 0; dNb = 0

        # State machine
        if book_state == 0:
            active_ra = ra_n; active_rb = rb_n
            book_state = 2; time_last_event = n
        elif book_state == 1:
            if outstanding_side == 'bid':
                if random.random() < alpha_t * math.exp(-k * (s[n] - active_rb)):
                    dNb = 1; book_state = 0
                elif n - time_last_event >= waiting_steps:
                    book_state = 0
            else:
                if random.random() < alpha_t * math.exp(-k * (active_ra - s[n])):
                    dNa = 1; book_state = 0
                elif n - time_last_event >= waiting_steps:
                    book_state = 0
        elif book_state == 2:
            if random.random() < alpha_t * math.exp(-k * (active_ra - s[n])): dNa = 1
            if random.random() < alpha_t * math.exp(-k * (s[n] - active_rb)):  dNb = 1
            if dNa and dNb:   book_state = 0
            elif dNa:         book_state = 1; outstanding_side = 'bid'; time_last_event = n
            elif dNb:         book_state = 1; outstanding_side = 'ask'; time_last_event = n
            elif n - time_last_event >= update_steps:
                active_ra = ra_n; active_rb = rb_n; time_last_event = n

        # Phi model + Gamma fill
        if   q[n] < 0: phi_bid = phi_max; phi_ask = phi_max * math.exp(-eta * abs(q[n]))
        elif q[n] > 0: phi_bid = phi_max * math.exp(-eta * q[n]); phi_ask = phi_max
        else:          phi_bid = phi_max; phi_ask = phi_max

        exec_ask = phi_ask * min(np.random.gamma(kappa, theta), 1.0) if dNa else 0
        exec_bid = phi_bid * min(np.random.gamma(kappa, theta), 1.0) if dNb else 0
        if dNa or dNb: trades += 1

        q[n+1]   = q[n] - exec_ask + exec_bid
        x[n+1]   = x[n] + active_ra*exec_ask - active_rb*exec_bid
        pnl[n+1] = x[n+1] + q[n+1]*s[n]

    return pnl, q, trades

##########################################
#   Monte Carlo loop
##########################################

for sigma in sigma_values:
    s_path = np.empty(N+1)
    for i_sim in range(n_sim):
        # One shared price path per simulation
        s_path[0] = 100
        bm.brownian(s_path[0], N, dt, sigma, out=s_path[1:])

        for st in ('opt', 'base'):
            pnl, q, trades = run_strategy(s_path, sigma, st)
            res = results[sigma][st]
            res['pnl_term'][i_sim] = pnl[-1]
            res['q_term'][i_sim]   = q[-1]
            res['n_trades'][i_sim] = trades

            pnl_arr = pnl[:-1]; q_arr = q[:-1]
            d = pnl_arr - res['pnl_mean']; res['pnl_mean'] += d/(i_sim+1); res['pnl_M2'] += d*(pnl_arr - res['pnl_mean'])
            d = q_arr   - res['q_mean'];   res['q_mean']   += d/(i_sim+1); res['q_M2']   += d*(q_arr   - res['q_mean'])

    for st in ('opt', 'base'):
        res = results[sigma][st]
        res['pnl_std'] = np.sqrt(res['pnl_M2'] / n_sim)
        res['q_std']   = np.sqrt(res['q_M2']   / n_sim)
    print(f"Done: sigma={sigma}")

##########################################
#   Summary table
##########################################

print(f"\n{'='*74}")
print(f"{'Scenario':<22} {'Strategy':<10} {'Avg PnL':>8} {'Std PnL':>8} {'Sharpe':>8} {'Avg Trades':>12}")
print(f"{'='*74}")
for sigma, label in zip(sigma_values, sigma_labels):
    for st, st_lbl in (('opt','Optimal'), ('base','Baseline')):
        res = results[sigma][st]
        avg = np.mean(res['pnl_term']); std = np.std(res['pnl_term'])
        print(f"{label:<22} {st_lbl:<10} {avg:>8.2f} {std:>8.2f} {avg/std if std>0 else 0:>8.3f} {np.mean(res['n_trades']):>12.1f}")
print(f"{'='*74}\n")

##########################################
#   Plots
##########################################

# ── 1. Mean PnL path ± 1σ band ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle('Mean P&L path ± 1σ  —  Optimal vs Baseline', fontsize=13)
for ax, sigma, label in zip(axes, sigma_values, sigma_labels):
    for st, col, name in (('opt', c_opt, 'Optimal'), ('base', c_base, 'Baseline')):
        m, s = results[sigma][st]['pnl_mean'], results[sigma][st]['pnl_std']
        ax.plot(t_plot, m, color=col, label=name)
        ax.fill_between(t_plot, m-s, m+s, alpha=0.2, color=col)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(label, fontsize=11); ax.set_xlabel('Time'); ax.set_ylabel('PnL [USD]')
    ax.grid(True); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig('../img/fig1_pnl_path.pdf', bbox_inches='tight'); plt.close()

# ── 2. Mean inventory path ± 1σ band ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle('Mean Inventory path ± 1σ  —  Optimal vs Baseline', fontsize=13)
for ax, sigma, label in zip(axes, sigma_values, sigma_labels):
    for st, col, name in (('opt', c_opt, 'Optimal'), ('base', c_base, 'Baseline')):
        m, s = results[sigma][st]['q_mean'], results[sigma][st]['q_std']
        ax.plot(t_plot, m, color=col, label=name)
        ax.fill_between(t_plot, m-s, m+s, alpha=0.2, color=col)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(label, fontsize=11); ax.set_xlabel('Time'); ax.set_ylabel('Inventory')
    ax.grid(True); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig('../img/fig2_inv_path.pdf', bbox_inches='tight'); plt.close()

# ── 3. Terminal PnL distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Terminal P&L distribution  —  Optimal vs Baseline', fontsize=13)
for ax, sigma, label in zip(axes, sigma_values, sigma_labels):
    ax.hist(results[sigma]['opt']['pnl_term'],  bins=40, alpha=0.6, color=c_opt,  label='Optimal')
    ax.hist(results[sigma]['base']['pnl_term'], bins=40, alpha=0.6, color=c_base, label='Baseline')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_title(label, fontsize=11); ax.set_xlabel('Terminal PnL'); ax.set_ylabel('Frequency')
    ax.grid(True); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig('../img/fig3_pnl_dist.pdf', bbox_inches='tight'); plt.close()

# ── 4. Terminal inventory distribution ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Terminal Inventory distribution  —  Optimal vs Baseline', fontsize=13)
for ax, sigma, label in zip(axes, sigma_values, sigma_labels):
    ax.hist(results[sigma]['opt']['q_term'],  bins=40, alpha=0.6, color=c_opt,  label='Optimal')
    ax.hist(results[sigma]['base']['q_term'], bins=40, alpha=0.6, color=c_base, label='Baseline')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_title(label, fontsize=11); ax.set_xlabel('Terminal Inventory'); ax.set_ylabel('Frequency')
    ax.grid(True); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig('../img/fig4_inv_dist.pdf', bbox_inches='tight'); plt.close()

# ── 5. PnL conditioned on terminal inventory ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('P&L conditioned on terminal inventory  |q|<1 (flat) vs |q|≥1 (exposed)', fontsize=12)
for ax, sigma, label in zip(axes, sigma_values, sigma_labels):
    for st, col, name in (('opt', c_opt, 'Optimal'), ('base', c_base, 'Baseline')):
        pnl_t = results[sigma][st]['pnl_term']
        q_t   = results[sigma][st]['q_term']
        flat    = pnl_t[np.abs(q_t) <  1]
        exposed = pnl_t[np.abs(q_t) >= 1]
        ax.hist(flat,    bins=30, alpha=0.5, color=col,       label=f'{name} flat')
        ax.hist(exposed, bins=30, alpha=0.3, color=col, hatch='//', label=f'{name} exposed')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_title(label, fontsize=11); ax.set_xlabel('Terminal PnL'); ax.set_ylabel('Frequency')
    ax.grid(True); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig('../img/fig5_pnl_cond.pdf', bbox_inches='tight'); plt.close()

# ── 6. Boxplot: PnL by scenario ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(sigma_values)); w = 0.3
bp1 = ax.boxplot([results[s]['opt']['pnl_term']  for s in sigma_values],
                 positions=pos-w/2, widths=0.25, patch_artist=True,
                 boxprops=dict(facecolor=c_opt,  alpha=0.8), medianprops=dict(color='black'))
bp2 = ax.boxplot([results[s]['base']['pnl_term'] for s in sigma_values],
                 positions=pos+w/2, widths=0.25, patch_artist=True,
                 boxprops=dict(facecolor=c_base, alpha=0.8), medianprops=dict(color='black'))
ax.set_xticks(pos); ax.set_xticklabels(sigma_labels, fontsize=11)
ax.set_ylabel('Terminal PnL [USD]', fontsize=13)
ax.set_title('Terminal P&L by Volatility Scenario', fontsize=13)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Optimal', 'Baseline'], fontsize=11)
ax.grid(True, axis='y'); plt.tight_layout(); plt.savefig('../img/fig6_boxplot.pdf', bbox_inches='tight'); plt.close()

# ── 7. Optimal spread over time (analytical) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
t_arr = np.linspace(0, T, 300)
for sigma, label in zip(sigma_values, sigma_labels):
    spread = gamma * sigma**2 * (T - t_arr) + 2/gamma * math.log(1 + gamma/k)
    ax.plot(t_arr, spread, label=label)
ax.set_xlabel('Time'); ax.set_ylabel('Optimal spread [USD]')
ax.set_title('A-S Optimal Spread vs. Time (γ=0.1, k=1.5)')
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig('../img/fig7_spread.pdf', bbox_inches='tight'); plt.close()

# ── 8. Intraday α(t) shape ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
t_arr = np.linspace(0, T, 300)
alpha_arr = np.array([get_alpha(t, T) for t in t_arr])
ax.plot(t_arr, alpha_arr, color='steelblue')
ax.set_xlabel('Time (fraction of day)'); ax.set_ylabel('α(t) — order arrival intensity')
ax.set_title('U-shaped Intraday Order Arrival Intensity')
ax.grid(True)
plt.tight_layout(); plt.savefig('../img/fig8_alpha.pdf', bbox_inches='tight'); plt.close()

print("All figures saved to ../img/")
