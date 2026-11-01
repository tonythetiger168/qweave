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
