"""Learned YAM workspace: one MLP forward pass answers "can the arm reach this pose?".

Full IK answers that per pose, iteratively, from a seed. A demo-placement search,
a check over every action in a policy chunk or live feedback while collecting
needs thousands of answers per frame, so this distils converged multi-start IK
into a small network (a learned workspace head):

    tool0 pose in the base frame (position + 6D rotation)
      -> p(reachable within 10 mm / 5 deg)
      -> best achievable position / rotation error
      -> a joint seed (the IK solution seeded from the nearest FK-table configs)

Commands (run from ~/karma, which has mujoco, mink and torch):
    workspace_model.py gen   --out artifacts/workspace-mlp/labels.npz --n 300000
    workspace_model.py train --labels artifacts/workspace-mlp/labels.npz --out artifacts/workspace-mlp/model.pt
    workspace_model.py eval  --model artifacts/workspace-mlp/model.pt --filtered artifacts/session-20261008-training/filtered-mink --out artifacts/workspace-mlp
    workspace_model.py fit   --model artifacts/workspace-mlp/model.pt --filtered artifacts/session-20261008-training/filtered-mink

Kinematics only: no self- or table-collision is modelled.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

POS_OK_M, ROT_OK_RAD = .010, np.radians(5.)
READY = np.array([0., .7, .7, 0., 0., 0.])
REST = np.array([0., np.pi/2, np.pi/2, 0., 0., 0.])
# Box the labels are drawn from (base frame, m). Max reach of tool0 is ~0.81 m.
BOX_LO, BOX_HI = np.array([-.6, -.9, -.25]), np.array([1.0, .9, 1.0])


# ---------------------------------------------------------------- features
def rot6d(R):
    """First two columns of R, the continuous 6D rotation encoding (Zhou et al. 2019)."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def features(T):
    T = np.asarray(T, float)
    return np.concatenate([T[..., :3, 3], rot6d(T[..., :3, :3])], -1).astype(np.float32)


# ---------------------------------------------------------------- converged IK (labels)
class ExactIK:
    """Multi-start, converged full-pose IK: the reference the network learns."""

    def __init__(self, iters=120, seeds=16, neighbours=8, table=200000, rng=None):
        import mink
        from mink_ik import MinkIKSolver
        self.mink = mink
        base = MinkIKSolver()  # reuse karma's model with the tool0 site
        self.model, self.data, self.site_id = base.model, base.data, base.site_id
        self.limits_q = base.joint_limits
        self.conf = mink.Configuration(self.model)
        self.task = mink.FrameTask('tool0', 'site', position_cost=1., orientation_cost=.3, lm_damping=.05)
        self.max_step = .15  # rad per iteration
        post = mink.PostureTask(self.model, cost=1e-4)
        rest = np.zeros(self.model.nq); rest[:6] = REST
        post.set_target(rest)
        self.tasks = [self.task, post]
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.iters, self.seeds, self.neighbours = iters, seeds, neighbours
        self.rng = rng or np.random.default_rng(0)
        # Random seeds alone miss narrow branches (folded elbow against the joint-4
        # stop: 35 deg off from 16 random starts, 2 mm / 3 deg from a nearby config).
        # Seed first from the joint configs whose FK is nearest the target.
        from scipy.spatial import cKDTree
        trng = np.random.default_rng(12345)  # same table in every worker
        self.table_q = trng.uniform(self.limits_q[:, 0], self.limits_q[:, 1], (table, 6))
        self.tree = cKDTree(np.stack([self._key(self.fk(q)) for q in self.table_q]))

    @staticmethod
    def _key(T):
        # 10 cm of position ~ 0.1 of a rotation-matrix column (~6 deg)
        return np.r_[T[:3, 3], .1*T[:3, 0], .1*T[:3, 1], .1*T[:3, 2]]

    def fk(self, q6):
        import mujoco
        self.data.qpos[:] = 0
        self.data.qpos[:6] = q6
        mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    def errors(self, T, q6):
        A = self.fk(q6)
        dR = T[:3, :3] @ A[:3, :3].T
        ang = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))
        return float(np.linalg.norm(T[:3, 3]-A[:3, 3])), float(ang)

    def solve_from(self, T, seed, max_step=None):
        """Damped, step-limited descent; returns the best iterate, early on entering the box.

        Full Newton steps (low damping, 0.5 rad) left even a seed sitting on a
        2 mm / 3 deg solution and ended 15 cm away, so steps are capped and the
        best iterate, not the last, is kept.
        """
        mink = self.mink
        q = np.zeros(self.model.nq); q[:6] = seed
        self.conf.update(q)
        self.task.set_target(mink.SE3.from_matrix(T))
        pe, re = self.errors(T, q[:6])
        best = (pe/POS_OK_M + re/ROT_OK_RAD, q[:6].copy(), pe, re)
        trace = []
        for it in range(self.iters):
            if best[2] <= POS_OK_M and best[3] <= ROT_OK_RAD and (best[2] < 5e-4 or it > 0):
                break
            v = mink.solve_ik(self.conf, self.tasks, 1., 'daqp', damping=1e-6, limits=self.limits)
            v[6:] = 0
            n = np.abs(v[:6]).max()
            self.conf.integrate_inplace(v*min(1., (max_step or self.max_step)/max(n, 1e-12)), 1.)
            q6 = self.conf.q[:6].copy()
            pe, re = self.errors(T, q6)
            score = pe/POS_OK_M + re/ROT_OK_RAD
            if score < best[0]:
                best = (score, q6, pe, re)
            # Stalled in a local minimum or against a limit: the next seed is cheaper.
            trace.append(best[0])
            if it >= 20 and trace[-20]-trace[-1] < 1e-3*trace[-1]:
                break
        return best[1], best[2], best[3]

    def __call__(self, T):
        """Returns (q6, pos_err_m, rot_err_rad, seeds_tried). Seed order is fixed: the
        nearest FK-table configs, then READY, REST, then random."""
        best = None
        _, near = self.tree.query(self._key(T), self.neighbours)
        seeds = [*self.table_q[near], READY, REST] + [self.rng.uniform(self.limits_q[:, 0], self.limits_q[:, 1])
                                                      for _ in range(self.seeds-2)]
        for k, s in enumerate(seeds):
            # Long steps from random seeds explore other branches; careful steps near home.
            q6, pe, re = self.solve_from(T, s, .5 if k >= self.neighbours+2 else None)
            score = pe/POS_OK_M + re/ROT_OK_RAD
            if best is None or score < best[3]:
                best = (q6, pe, re, score)
            if pe <= POS_OK_M and re <= ROT_OK_RAD:
                return q6, pe, re, k+1
        # The weighted optimum can sit outside the 10 mm / 5 deg box while another
        # trade-off is inside it (at a joint limit: 10 mm / 1.6 deg vs 2.6 mm / 3.5 deg).
        # Re-polish the best answer at other rotation weights before calling it unreachable.
        try:
            for w in (.01, .1, 1., 3.):
                self.task.set_orientation_cost(w)
                q6, pe, re = self.solve_from(T, best[0])
                if pe <= POS_OK_M and re <= ROT_OK_RAD:
                    return q6, pe, re, len(seeds)+1
                score = pe/POS_OK_M + re/ROT_OK_RAD
                if score < best[3]:
                    best = (q6, pe, re, score)
        finally:
            self.task.set_orientation_cost(.3)
        return best[0], best[1], best[2], len(seeds)+1


def random_rotation(rng, n):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=rng.integers(1 << 31)).as_matrix()


def sample_targets(ik, rng, n):
    """Half near the reachable set (FK of random joints, then perturbed), half a box."""
    from scipy.spatial.transform import Rotation
    T = np.tile(np.eye(4), (n, 1, 1))
    kind = rng.choice(3, n, p=[.25, .35, .40])
    for i in range(n):
        if kind[i] == 2:
            T[i, :3, 3] = rng.uniform(BOX_LO, BOX_HI)
            T[i, :3, :3] = random_rotation(rng, 1)[0]
            continue
        T[i] = ik.fk(rng.uniform(ik.limits_q[:, 0], ik.limits_q[:, 1]))
        if kind[i] == 1:  # push off the manifold to populate the boundary
            T[i, :3, 3] += rng.normal(0, .05, 3)
            T[i, :3, :3] = Rotation.from_rotvec(rng.normal(0, .35, 3)).as_matrix() @ T[i, :3, :3]
    return T, kind


def _gen_worker(args):
    seed, n = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rng = np.random.default_rng(seed)
    ik = ExactIK(rng=rng)
    T, kind = sample_targets(ik, rng, n)
    out = np.zeros((n, 9))
    for i in range(n):
        q6, pe, re, k = ik(T[i])
        out[i] = [*q6, pe, re, k]
    return T, kind, out


def cmd_gen(a):
    from multiprocessing import Pool
    jobs = [(a.seed*1000+j, a.n//a.workers) for j in range(a.workers)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        parts = pool.map(_gen_worker, jobs)
    T = np.concatenate([p[0] for p in parts]); kind = np.concatenate([p[1] for p in parts])
    lab = np.concatenate([p[2] for p in parts])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, T=T, kind=kind, q=lab[:, :6], pos_err=lab[:, 6], rot_err=lab[:, 7], seeds=lab[:, 8])
    reach = (lab[:, 6] <= POS_OK_M) & (lab[:, 7] <= ROT_OK_RAD)
    print(json.dumps({'n': len(T), 'seconds': round(time.time()-t0, 1), 'reachable': float(reach.mean()),
                      'reachable_by_kind': [float(reach[kind == k].mean()) for k in range(3)]}))


# ---------------------------------------------------------------- network
def _torch_net(hidden=256, depth=4):
    from torch import nn
    layers, d = [], 9
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    trunk = nn.Sequential(*layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.reach = nn.Linear(hidden, 1)
            self.err = nn.Linear(hidden, 2)   # log1p(pos_mm), log1p(rot_deg)
            self.seed = nn.Linear(hidden, 6)

        def forward(self, x):
            h = self.trunk(x)
            return self.reach(h)[..., 0], self.err(h), self.seed(h)
    return Net()


class WorkspaceModel:
    """numpy in, numpy out. `predict(T)` with T of shape (..., 4, 4)."""

    def __init__(self, path):
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.net = _torch_net(**ckpt['arch'])
        self.net.load_state_dict(ckpt['state'])
        self.net.eval()
        self.mu, self.sd = ckpt['mu'], ckpt['sd']
        self.meta = ckpt['meta']
        self.torch = torch

    def tensor(self, T):
        return self.torch.as_tensor((features(T)-self.mu)/self.sd)

    def predict(self, T):
        T = np.asarray(T, float)
        shape = T.shape[:-2]
        with self.torch.no_grad():
            logit, err, seed = self.net(self.tensor(T.reshape(-1, 4, 4)))
        err = np.expm1(err.numpy()).clip(0)
        return {'p_reach': (1/(1+np.exp(-logit.numpy()))).reshape(shape),
                'pos_err_mm': err[:, 0].reshape(shape), 'rot_err_deg': err[:, 1].reshape(shape),
                'q_seed': seed.numpy().reshape(*shape, 6)}


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    d = np.load(a.labels)
    X = features(d['T'])
    reach = ((d['pos_err'] <= POS_OK_M) & (d['rot_err'] <= ROT_OK_RAD)).astype(np.float32)
    err = np.log1p(np.stack([d['pos_err']*1000, np.degrees(d['rot_err'])], 1)).astype(np.float32)
    q = d['q'].astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nval = len(X)//10
    va, tr = idx[:nval], idx[nval:]
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-6
    Xn = torch.as_tensor((X-mu)/sd)
    R, E, Q = map(torch.as_tensor, (reach, err, q))
    net = _torch_net(a.hidden, a.depth)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-5)
    steps = a.epochs*(len(tr)//a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    torch.set_num_threads(a.threads)

    def loss_fn(ix):
        logit, e, s = net(Xn[ix])
        lr = F.binary_cross_entropy_with_logits(logit, R[ix])
        le = F.smooth_l1_loss(e, E[ix])
        # Seeds only matter where IK can actually get there.
        m = R[ix] > 0
        ls = F.smooth_l1_loss(s[m], Q[ix][m]) if m.any() else logit.sum()*0
        return lr+le+ls, (lr.item(), le.item(), ls.item())

    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr))
        for b in range(len(tr)//a.batch):
            loss, _ = loss_fn(perm[b*a.batch:(b+1)*a.batch])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            logit, _, _ = net(Xn[va])
            acc = ((logit > 0).float() == R[va]).float().mean().item()
            _, parts = loss_fn(torch.as_tensor(va))
        print(f'epoch {ep+1:3d}  val acc {acc:.4f}  bce/err/seed {parts[0]:.4f} {parts[1]:.4f} {parts[2]:.4f}  '
              f'{time.time()-t0:.0f}s', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': net.state_dict(), 'arch': {'hidden': a.hidden, 'depth': a.depth}, 'mu': mu, 'sd': sd,
                'meta': {'labels': str(a.labels), 'n_train': len(tr), 'val_idx_seed': 0, 'val_acc': acc,
                         'pos_ok_m': POS_OK_M, 'rot_ok_rad': float(ROT_OK_RAD)}}, a.out)


# ---------------------------------------------------------------- placement
def shift_transform(dx, dy, dz, dyaw_deg):
    """Base-frame correction D: rotate about base z by dyaw, then translate."""
    c, s = np.cos(np.radians(dyaw_deg)), np.sin(np.radians(dyaw_deg))
    D = np.eye(4)
    D[:2, :2] = [[c, -s], [s, c]]
    D[:3, 3] = dx, dy, dz
    return D


def shifted_anchor(anchor, shift):
    """Anchor (x, y, z, yaw) such that anchor_transform(new) = D @ anchor_transform(anchor)."""
    D = shift_transform(*shift)
    return [*(D[:3, :3] @ np.asarray(anchor[:3], float) + D[:3, 3]), float(anchor[3]+shift[3])]


def fit_placement(model, T, max_shift_m=.25, max_yaw_deg=45., steps=150, shift_cost=.5, max_frames=1500):
    """Move the whole demo (rigidly, about base z) so the arm can reach as much of it as possible.

    Gradient ascent through the network on mean log p(reachable) - which punishes
    the worst frames hardest - with a small cost per 10 cm moved so a demo that
    is already reachable stays put. Several starts; the best by predicted
    reachable fraction, then by distance moved, wins. Returns the shift
    (dx, dy, dz m, dyaw deg) and the predicted reachable fraction before and after.
    """
    torch = model.torch
    T = np.asarray(T, float)
    if len(T) > max_frames:
        T = T[np.linspace(0, len(T)-1, max_frames).astype(int)]
    p0 = torch.as_tensor(T[:, :3, 3], dtype=torch.float32)
    c0 = torch.as_tensor(np.concatenate([T[:, :3, 0], T[:, :3, 1]], 0), dtype=torch.float32)  # (2N, 3)
    mu, sd = torch.as_tensor(model.mu), torch.as_tensor(model.sd)
    for prm in model.net.parameters():
        prm.requires_grad_(False)

    def logits(x):
        dx, yaw = x[:3], x[3]
        Rz = torch.stack([torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros(())]),
                          torch.stack([torch.sin(yaw), torch.cos(yaw), torch.zeros(())]),
                          torch.tensor([0., 0., 1.])])
        p, cols = p0 @ Rz.T + dx, c0 @ Rz.T
        f = torch.cat([p, cols[:len(T)], cols[len(T):]], 1)
        return model.net((f-mu)/sd)[0]

    best = None
    starts = [(0, 0, 0, 0), (-.1, 0, 0, 0), (.1, 0, 0, 0), (0, -.1, 0, 0), (0, .1, 0, 0),
              (0, 0, 0, -25), (0, 0, 0, 25)]
    for start in starts:
        x = torch.tensor([*start[:3], np.radians(start[3])], dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([x], lr=.01)
        for _ in range(steps):
            obj = torch.nn.functional.logsigmoid(logits(x)).mean() - shift_cost*(x[:3].norm()/.1)**2*.01
            opt.zero_grad(); (-obj).backward(); opt.step()
            with torch.no_grad():
                x[:3].clamp_(-max_shift_m, max_shift_m)
                x[3].clamp_(-np.radians(max_yaw_deg), np.radians(max_yaw_deg))
        with torch.no_grad():
            frac = (logits(x) > 0).float().mean().item()
        cand = (frac, -x[:3].norm().item(), x.detach().numpy().copy())
        if best is None or cand[:2] > best[:2]:
            best = cand
    with torch.no_grad():
        before = (logits(torch.zeros(4)) > 0).float().mean().item()
    x = best[2]
    return [float(x[0]), float(x[1]), float(x[2]), float(np.degrees(x[3]))], before, best[0]


def cmd_fit(a):
    """One placement for the whole session (all episodes share the world anchor), checked by exact IK."""
    m = WorkspaceModel(a.model)
    demos = demo_targets(a.filtered)
    T = np.concatenate([e['T'] for e in demos.values()])
    t0 = time.perf_counter()
    shift, before, after = fit_placement(m, T)
    fit_s = time.perf_counter()-t0
    ik = ExactIK(rng=np.random.default_rng(3))
    D = shift_transform(*shift)
    report = {'shift_dx_dy_dz_m_dyaw_deg': [round(v, 4) for v in shift], 'fit_seconds': round(fit_s, 2),
              'apply_with': 'filter_training_ik.py --anchor-shift '+' '.join(f'{v:.4f}' for v in shift),
              'mlp_reachable_before': before, 'mlp_reachable_after': after, 'episodes': {}}
    def reach(Ts):
        return np.array([pe <= POS_OK_M and re <= ROT_OK_RAD for _, pe, re, _ in map(ik, Ts)])

    for name, e in demos.items():
        b, af = reach(e['T']), reach(D @ e['T'])
        report['episodes'][name] = {'frames': len(b), 'exact_unreachable_before': int((~b).sum()),
                                    'exact_unreachable_after': int((~af).sum())}
    report['exact_unreachable_before'] = sum(v['exact_unreachable_before'] for v in report['episodes'].values())
    report['exact_unreachable_after'] = sum(v['exact_unreachable_after'] for v in report['episodes'].values())
    print(json.dumps(report, indent=2))
    if a.out:
        a.out.write_text(json.dumps(report, indent=2)+'\n')


# ---------------------------------------------------------------- evaluation
def demo_targets(filtered):
    """All tool0 targets in the base frame from a filter_training_ik output, per episode."""
    out = {}
    for path in sorted(Path(filtered).glob('episode_*.json')):
        doc = json.loads(path.read_text())
        rows = [(i, np.asarray(f['T_robot_target'])) for i, f in enumerate(doc['frames']) if 'T_robot_target' in f]
        out[path.stem] = {'frame': np.array([r[0] for r in rows]), 'T': np.stack([r[1] for r in rows]),
                          'filter_ik_rejected': np.array(['IK tracking exceeds 10mm / 5deg' in doc['frames'][i]['reasons']
                                                          for i, _ in rows])}
    return out


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    pos = label.astype(bool)
    return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*(~pos).sum()))


def cmd_eval(a):
    m = WorkspaceModel(a.model)
    report = {}

    # 1. held-out random labels (same split as training)
    d = np.load(m.meta['labels'])
    idx = np.random.default_rng(0).permutation(len(d['T']))[:len(d['T'])//10]
    y = (d['pos_err'][idx] <= POS_OK_M) & (d['rot_err'][idx] <= ROT_OK_RAD)
    t0 = time.perf_counter(); p = m.predict(d['T'][idx]); dt = time.perf_counter()-t0
    pred = p['p_reach'] > .5
    report['heldout'] = {'n': len(idx), 'acc': float((pred == y).mean()), 'auc': auc(p['p_reach'], y),
                         'false_reachable': float((pred & ~y).mean()), 'false_unreachable': float((~pred & y).mean()),
                         'us_per_pose_batched': dt/len(idx)*1e6}

    # 2. the real demo frames, against converged IK computed here
    ik = ExactIK(rng=np.random.default_rng(1))
    demos = demo_targets(a.filtered)
    T = np.concatenate([e['T'] for e in demos.values()])
    filt = np.concatenate([e['filter_ik_rejected'] for e in demos.values()])
    ep = np.concatenate([[k]*len(e['T']) for k, e in demos.items()])
    t0 = time.perf_counter()
    lab = np.array([ik(t)[1:] for t in T]); ik_s = (time.perf_counter()-t0)/len(T)
    y = (lab[:, 0] <= POS_OK_M) & (lab[:, 1] <= ROT_OK_RAD)
    p = m.predict(T); pred = p['p_reach'] > .5
    report['demo_frames'] = {
        'n': len(T), 'exact_ik_unreachable': int((~y).sum()), 'mlp_unreachable': int((~pred).sum()),
        'filter_rejected': int(filt.sum()), 'acc_vs_exact_ik': float((pred == y).mean()),
        'auc_vs_exact_ik': auc(p['p_reach'], y) if 0 < y.sum() < len(y) else None,
        'mlp_missed_unreachable': int((pred & ~y).sum()), 'mlp_false_alarm': int((~pred & y).sum()),
        'acc_vs_filter': float((pred == ~filt).mean()),
        'exact_ik_ms_per_pose': ik_s*1e3,
        'by_episode': {k: {'n': int((ep == k).sum()), 'exact_unreachable': int((~y[ep == k]).sum()),
                           'mlp_unreachable': int((~pred[ep == k]).sum()),
                           'filter_rejected': int(filt[ep == k].sum())} for k in demos}}

    # 3. seed head: does it converge faster than READY on demo frames?
    sub = np.random.default_rng(2).choice(len(T), min(400, len(T)), replace=False)
    iters = {'ready': [], 'mlp': []}
    ok = {'ready': 0, 'mlp': 0}
    for i in sub:
        for name, s in (('ready', READY), ('mlp', np.clip(p['q_seed'][i], ik.limits_q[:, 0], ik.limits_q[:, 1]))):
            it, q = _iters_to_converge(ik, T[i], s)
            iters[name].append(it); ok[name] += q
    report['seed_head'] = {'frames': len(sub),
                           'median_iters': {k: float(np.median(v)) for k, v in iters.items()},
                           'converged': ok}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out/'demo_eval.npz', T=T, ep=ep, filt=filt, exact=lab, **{k: v for k, v in p.items()})
    (a.out/'eval.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def _iters_to_converge(ik, T, seed):
    mink = ik.mink
    q = np.zeros(ik.model.nq); q[:6] = seed
    ik.conf.update(q); ik.task.set_target(mink.SE3.from_matrix(T))
    for it in range(1, ik.iters+1):
        v = mink.solve_ik(ik.conf, ik.tasks, 1., 'daqp', damping=1e-6, limits=ik.limits)
        v[6:] = 0
        ik.conf.integrate_inplace(v*min(1., ik.max_step/max(np.abs(v[:6]).max(), 1e-12)), 1.)
        pe, re = ik.errors(T, ik.conf.q[:6])
        if pe <= POS_OK_M and re <= ROT_OK_RAD:
            return it, 1
    return ik.iters, 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen'); g.add_argument('--out', type=Path, required=True)
    g.add_argument('--n', type=int, default=300000); g.add_argument('--workers', type=int, default=max(1, os.cpu_count()-1))
    g.add_argument('--seed', type=int, default=0)
    t = sub.add_parser('train'); t.add_argument('--labels', type=Path, required=True); t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=40); t.add_argument('--batch', type=int, default=1024)
    t.add_argument('--hidden', type=int, default=256); t.add_argument('--depth', type=int, default=4)
    t.add_argument('--threads', type=int, default=8)
    e = sub.add_parser('eval'); e.add_argument('--model', type=Path, required=True)
    e.add_argument('--filtered', type=Path, required=True); e.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('fit'); f.add_argument('--model', type=Path, required=True)
    f.add_argument('--filtered', type=Path, required=True); f.add_argument('--out', type=Path)
    a = p.parse_args()
    {'gen': cmd_gen, 'train': cmd_train, 'eval': cmd_eval, 'fit': cmd_fit}[a.cmd](a)


if __name__ == '__main__':
    main()
