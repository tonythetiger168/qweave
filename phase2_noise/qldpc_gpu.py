"""
Phase E5-b: Batched BP+OSD decoder in PyTorch (GPU)
====================================================
Same algorithm as ``qldpc.BpOsdDecoder`` (normalized min-sum BP, each
shot frozen at its first converged iteration, OSD-0 for the rest), with
BP run on the GPU for thousands of syndromes at once.

Speed-ups over the numpy version:
  * messages stay in a check-major (S, m, wr) layout; the variable
    update is one gather from that layout, no scatter per iteration
  * min-sum uses min/argmin + second min instead of a full sort
  * sign product via parity of negative counts
  * converged shots are dropped from the working set every iteration
  * OSD-0 runs on CPU only for shots BP could not solve (a few % at
    practical error rates)
"""

from typing import Optional, Tuple
import numpy as np
import torch

from phase2_noise.qldpc import BpOsdDecoder


class TorchBpOsdDecoder:
    def __init__(self, H: np.ndarray, error_rate: float, max_iter: int = 100,
                 ms_scaling: float = 0.625, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32):
        self.cpu = BpOsdDecoder(H, error_rate, max_iter, ms_scaling)  # OSD + layout
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.max_iter = max_iter
        self.alpha = ms_scaling
        self.prior = float(self.cpu.prior)
        H = self.cpu.H
        self.m, self.n = H.shape
        dev = self.device

        chk = self.cpu.chk                         # (m, wr) edge ids, -1 pad
        self.wr = chk.shape[1]
        self.chk_mask = torch.as_tensor(chk >= 0, device=dev)
        # variable of each check slot (pad -> n, a dummy always-zero column)
        col_of_edge = self.cpu.edge_col
        slot_var = np.where(chk >= 0, col_of_edge[np.maximum(chk, 0)], self.n)
        self.slot_var = torch.as_tensor(slot_var.reshape(-1), device=dev)
        # check-slot position of each variable's edges (pad -> dummy slot m*wr)
        pos_of_edge = np.empty(self.cpu.E, np.int64)
        r, c = np.nonzero(chk >= 0)
        pos_of_edge[chk[r, c]] = r * self.wr + c
        var = self.cpu.var
        var_slot = np.where(var >= 0, pos_of_edge[np.maximum(var, 0)], self.m * self.wr)
        self.var_slot = torch.as_tensor(var_slot, device=dev)          # (n, wc)

    @torch.no_grad()
    def bp(self, syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dev, dt = self.device, self.dtype
        S = syndromes.shape[0]
        synd = torch.as_tensor(syndromes, device=dev, dtype=torch.bool)
        m, n, wr = self.m, self.n, self.wr
        mask = self.chk_mask.unsqueeze(0)
        inf = torch.tensor(float("inf"), device=dev, dtype=dt)

        hard_out = torch.zeros((S, n), dtype=torch.uint8, device=dev)
        post_out = torch.full((S, n), self.prior, dtype=dt, device=dev)
        done = torch.zeros(S, dtype=torch.bool, device=dev)
        active = torch.arange(S, device=dev)
        s_act = synd
        v2c = torch.full((S, m, wr), self.prior, dtype=dt, device=dev)

        for _ in range(self.max_iter):
            # ---- check update (min-sum, excluding self) ----
            mag = torch.where(mask, v2c.abs(), inf)
            neg = (v2c < 0) & mask
            parity = (neg.sum(2) % 2).bool() ^ s_act                 # (A, m)
            min1, arg1 = mag.min(2)
            mag2 = mag.scatter(2, arg1.unsqueeze(2), float("inf"))
            min2 = mag2.min(2).values
            is_min = torch.arange(wr, device=dev).view(1, 1, wr) == arg1.unsqueeze(2)
            out_mag = torch.where(is_min, min2.unsqueeze(2), min1.unsqueeze(2)) * self.alpha
            out_neg = parity.unsqueeze(2) ^ neg
            c2v = torch.where(out_neg, -out_mag, out_mag)
            c2v = torch.where(mask, c2v, torch.zeros((), device=dev, dtype=dt))

            # ---- variable update ----
            flat = torch.cat([c2v.reshape(c2v.shape[0], -1),
                              torch.zeros((c2v.shape[0], 1), device=dev, dtype=dt)], 1)
            inc = flat[:, self.var_slot]                              # (A, n, wc)
            post = self.prior + inc.sum(2)                            # (A, n)
            post_pad = torch.cat([post, torch.zeros((post.shape[0], 1), device=dev, dtype=dt)], 1)
            v2c = (post_pad[:, self.slot_var] - flat[:, :-1]).view(-1, m, wr)

            # ---- convergence: freeze solved shots, drop them ----
            cur = post < 0
            cur_pad = torch.cat([cur, torch.zeros((cur.shape[0], 1), device=dev, dtype=torch.bool)], 1)
            got = (cur_pad[:, self.slot_var].view(-1, m, wr).sum(2) % 2).bool()
            ok = (got == s_act).all(1)
            hard_out[active] = cur.to(torch.uint8)
            post_out[active] = post
            if ok.any():
                done[active[ok]] = True
                keep = ~ok
                active, s_act, v2c = active[keep], s_act[keep], v2c[keep]
                if active.numel() == 0:
                    break
        return (hard_out.cpu().numpy(), post_out.double().cpu().numpy(),
                done.cpu().numpy())

    @torch.no_grad()
    def osd0_batch(self, syndromes: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        """
        OSD-0 for many shots at once: GF(2) elimination of H with each
        shot's own column order (most likely flipped first), run as
        one tensor op per column over all shots.
        """
        dev = self.device
        F = syndromes.shape[0]
        H = torch.as_tensor(self.cpu.H, device=dev, dtype=torch.bool)
        order = torch.argsort(posterior, dim=1, stable=True)            # (F, n)
        A = torch.cat([H[:, order].permute(1, 0, 2),                      # (F, m, n)
                       syndromes.bool().unsqueeze(2)], dim=2)           # (F, m, n+1)
        used = torch.zeros((F, self.m), dtype=torch.bool, device=dev)
        piv_row = torch.full((F, self.n), -1, dtype=torch.long, device=dev)
        rows = torch.arange(self.m, device=dev)
        bidx = torch.arange(F, device=dev)
        n_piv = torch.zeros(F, dtype=torch.long, device=dev)
        for j in range(self.n):
            col = A[:, :, j]
            cand = col & ~used
            has = cand.any(1)
            if not has.any():
                if bool((n_piv >= self.cpu.rank).all()):
                    break
                continue
            r = cand.to(torch.uint8).argmax(1)                            # first candidate row
            pivot = A[bidx, r]                                            # (F, n+1)
            elim = col & (rows.unsqueeze(0) != r.unsqueeze(1)) & has.unsqueeze(1)
            A ^= elim.unsqueeze(2) & pivot.unsqueeze(1)
            used[bidx[has], r[has]] = True
            piv_row[has, j] = r[has]
            n_piv += has.long()
            if bool((n_piv >= self.cpu.rank).all()):
                break
        sol = torch.zeros((F, self.n), dtype=torch.uint8, device=dev)
        b, j = torch.nonzero(piv_row >= 0, as_tuple=True)
        vals = A[b, piv_row[b, j], self.n].to(torch.uint8)
        sol[b, order[b, j]] = vals
        return sol

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, np.uint8)
        hard, post, done = self.bp(syndromes)
        fail = np.nonzero(~done)[0]
        if len(fail):
            s = torch.as_tensor(syndromes[fail], device=self.device)
            pst = torch.as_tensor(post[fail], device=self.device)
            hard[fail] = self.osd0_batch(s, pst).cpu().numpy()
        return hard


class RelayBpDecoder:
    """
    Relay-BP (Mueller et al., IBM 2025): a chain of memory-BP "legs".
    Each leg runs min-sum BP whose variable bias mixes the prior with the
    previous marginal, Lambda_j(t) = (1-g_j) Lambda_j(0) + g_j M_j(t-1);
    leg 0 uses g = gamma0, later legs draw g_j uniformly in
    [gamma_min, gamma_max] (negative values allowed) and start from the
    last marginals. Every converged leg yields a candidate; the decoder
    returns the lowest-weight one after ``solutions`` are found or
    ``legs`` run out. No OSD stage.

    Works on any parity-check matrix with per-column priors, e.g. the
    detector error model of a syndrome circuit.
    """

    def __init__(self, H: np.ndarray, priors: np.ndarray, legs: int = 301,
                 first_iters: int = 80, leg_iters: int = 60, solutions: int = 5,
                 gamma0: float = 0.125, gamma_range=(-0.24, 0.66),
                 ms_scaling: float = 1.0, device: Optional[str] = None, seed: int = 0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        H = (np.asarray(H) & 1).astype(np.uint8)
        self.m, self.n = H.shape
        self.legs, self.T0, self.Tr, self.S = legs, first_iters, leg_iters, solutions
        self.g0, self.grange, self.alpha = gamma0, gamma_range, ms_scaling
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        dev = self.device
        pr = np.clip(np.asarray(priors, np.float64), 1e-12, 0.5 - 1e-12)
        self.llr0 = torch.as_tensor(np.log((1 - pr) / pr), device=dev, dtype=torch.float32)

        rows, cols = np.nonzero(H)
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        wr = int(np.bincount(rows, minlength=self.m).max())
        slot = np.arange(len(rows)) - np.searchsorted(rows, rows)        # position in its row
        pos = rows * wr + slot
        self.wr = wr
        mask = np.zeros(self.m * wr, bool); mask[pos] = True
        slot_var = np.full(self.m * wr, self.n); slot_var[pos] = cols
        by_col = np.argsort(cols, kind="stable")
        wc = int(np.bincount(cols, minlength=self.n).max())
        cslot = np.arange(len(cols)) - np.searchsorted(cols[by_col], cols[by_col])
        var_slot = np.full((self.n, wc), self.m * wr)
        var_slot[cols[by_col], cslot] = pos[by_col]
        self.chk_mask = torch.as_tensor(mask.reshape(self.m, wr), device=dev)
        self.slot_var = torch.as_tensor(slot_var, device=dev)
        self.var_slot = torch.as_tensor(var_slot, device=dev)
        self.Hd = torch.as_tensor(H, device=dev)

    @torch.no_grad()
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        dev = self.device
        S0 = syndromes.shape[0]
        m, n, wr = self.m, self.n, self.wr
        synd_all = torch.as_tensor(np.asarray(syndromes), device=dev).bool()
        mask = self.chk_mask.unsqueeze(0)
        inf = float("inf")
        zcol = lambda A: torch.zeros((A, 1), device=dev)

        best = torch.zeros((S0, n), dtype=torch.uint8, device=dev)
        best_w = torch.full((S0,), inf, device=dev)
        found = torch.zeros(S0, dtype=torch.long, device=dev)
        last = torch.zeros((S0, n), dtype=torch.uint8, device=dev)

        active = torch.arange(S0, device=dev)
        s_act = synd_all
        M = self.llr0.expand(S0, n).clone()                   # marginals
        for leg in range(self.legs):
            A = active.numel()
            if A == 0:
                break
            if leg == 0:
                g = torch.full((A, n), self.g0, device=dev)
                T = self.T0
            else:
                lo, hi = self.grange
                g = lo + (hi - lo) * torch.rand((A, n), generator=self.gen, device=dev)
                T = self.Tr
            v2c = torch.cat([M, zcol(A)], 1)[:, self.slot_var].view(A, m, wr)
            running = torch.ones(A, dtype=torch.bool, device=dev)
            for _ in range(T):
                mag = torch.where(mask, v2c.abs(), torch.tensor(inf, device=dev))
                neg = (v2c < 0) & mask
                parity = (neg.sum(2) % 2).bool() ^ s_act
                min1, arg1 = mag.min(2)
                min2 = mag.scatter(2, arg1.unsqueeze(2), inf).min(2).values
                is_min = torch.arange(wr, device=dev).view(1, 1, wr) == arg1.unsqueeze(2)
                out = torch.where(is_min, min2.unsqueeze(2), min1.unsqueeze(2)) * self.alpha
                c2v = torch.where(parity.unsqueeze(2) ^ neg, -out, out)
                c2v = torch.where(mask, c2v, torch.zeros((), device=dev))
                flat = torch.cat([c2v.reshape(A, -1), zcol(A)], 1)
                bias = (1 - g) * self.llr0 + g * M
                post = bias + flat[:, self.var_slot].sum(2)
                # shots that already converged in this leg keep their state
                M = torch.where(running.unsqueeze(1), post, M)
                v2c_new = (torch.cat([post, zcol(A)], 1)[:, self.slot_var] - flat[:, :-1]).view(A, m, wr)
                v2c = torch.where(running.view(A, 1, 1), v2c_new, v2c)
                hard = (M < 0)
                got = (torch.cat([hard, torch.zeros((A, 1), dtype=torch.bool, device=dev)], 1)
                       [:, self.slot_var].view(A, m, wr).sum(2) % 2).bool()
                ok = (got == s_act).all(1) & running
                if ok.any():
                    idx = active[ok]
                    e = hard[ok].to(torch.uint8)
                    w = (e.float() * self.llr0).sum(1)
                    better = w < best_w[idx]
                    best[idx[better]] = e[better]
                    best_w[idx[better]] = w[better]
                    found[idx] += 1
                    running &= ~ok
                if not running.any():
                    break
            last[active] = (M < 0).to(torch.uint8)
            keep = found[active] < self.S
            active, s_act, M = active[keep], s_act[keep], M[keep]
        out = torch.where((best_w < inf).unsqueeze(1), best, last)
        return out.cpu().numpy()
