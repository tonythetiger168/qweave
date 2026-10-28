# 超越 FermiQLLM 1.0 — 改進計畫 (Surpass Plan)

> 2026-10-28 · companion to [QUANTUM_LLM_ROADMAP.md](QUANTUM_LLM_ROADMAP.md),
> [FERMIQLLM_GLOBAL_INVESTIGATION.md](FERMIQLLM_GLOBAL_INVESTIGATION.md),
> measured results in [QL_RESULTS.md](QL_RESULTS.md).

## 0. 一句話

FermiQLLM 1.0 的五個「量子增強」環節都只是公司內測數字；qweave 每個環節都要做到
**可下載、可重跑、對 dense 基線比較、含負面結果**，再在公開榜單上用同等算力超過它。

## 1. 逐環節對照：他們聲稱什麼 → 我們已交付什麼 → 超越目標

| 環節 (FermiQLLM C1) | FermiQLLM 1.0 主張 | qweave 已交付 (本 PR) | 超越目標 (可驗證) |
|---|---|---|---|
| 模型結構 | 張量網絡嵌入，推理 +15% | TTLinear（交錯式 MPO）+ 全局參數預算；修正舊 MPO 把矩陣秩卡在 χ 的缺陷 | 同參數量下 PPL / 任務分數公開對比；Triton 融合核 ≥1.3× 解碼吞吐 |
| 量子模擬退火 | 「量子退火」用於優化 | 路徑積分 SQA（Trotter 複本 + 橫場退火）求逐層秩分配，對 exact DP 量測最優性差距 | 分配器差距 <1% vs 精確解；擴展到 RL 更新選擇 (QL2b) |
| 訓練 / 修復 | 訓練成本下降 | 知識蒸餾 heal（KL 對 dense 老師）+ fp32 主權重 bf16 autocast；報告恢復率 | heal 後 PPL ≤ dense +3%，參數 −30% |
| 強化學習 | RL 成本 −25% | （下一步）GRPO + 壓縮策略 | 同 GPU-hours 下 GSM8K/MATH-500 ≥ dense |
| 評測 | MATH-500/GPQA/BBH +10–20%（內測） | lm-eval-harness 對比 + 自定魯棒性套件（字元噪聲、長上下文漂移），全部 JSON 公開 | 第三方可一鍵復現；公開榜單提交 |
| 注意力（他們未提） | — | QSA：以注意力頭的糾纏熵分配 block-sparse 預算（flex_attention） | 16K+ 上下文 ≥1.2× 注意力吞吐，PPL 增幅 ≤1% |

## 2. 實測結果摘要

(filled from QL_RESULTS.md after the RunPod run)

## 3. 改進計畫（QL5–QL8）

### QL5 — 規模化到 FermiQLLM 同級基座（Qwen3-4B / 8B）
- 目標：與 FermiQLLM 同一 Qwen 基座家族，報告 0.6B → 4B → 8B 的壓縮曲線
  （參數節省 vs PPL / MMLU / GSM8K / BBH / GPQA-Diamond / MATH-500）。
- 手段：逐層敏感度 = 激活加權誤差（‖(W−Ŵ)X‖ 而非 ‖W−Ŵ‖），SQA 分配器；
  heal 用 ≥50M token 蒸餾，FSDP 4×H100。
- 驗收：−30% 參數時 MMLU 下降 ≤1pt；全部權重、配置、log 開源至 HF。

### QL6 — 速度真的變快：融合 TT 核 + 量化疊加
- Triton 融合 TT-matmul（核心常駐 SRAM，不實體化 W）；decode 階段 batch=1 是 TN
  的最大優勢場景（記憶體頻寬受限）。
- TT 核 + INT8/FP8 量化疊加，對比 GPTQ/AWQ 單獨量化的 Pareto 前沿。
- 驗收：vLLM 整合後 decode tokens/s ≥1.3× dense（FermiQLLM 宣稱 +15%）。

### QL7 — 「量子退火」用在真正的組合問題上
- RL：GRPO 中把「哪些層 / 哪些秩方向更新」寫成 QUBO，SQA 選擇；
  量測 GPU-hours vs 全量 AdamW（FermiQLLM 宣稱 −25%）。
- 資料：用 SQA 做訓練資料子集選擇（覆蓋度 − 冗餘的 QUBO）。
- 驗收：同等最終分數下 GPU-hours ≤0.75×，並公布失敗的配置。

### QL8 — 量子硬體接口（保留原始使命）
- 本套件的 Clifford / DEM / PyMatching 管線（Phase II 已修復）產生器件級錯誤模型；
  在 FermiQLLM 未觸及的方向——小規模 QPU 上的 kernel / 取樣子程序——留下可驗證接口。
- 誠實定位：在可證明量子優勢出現前，一律稱 *quantum-inspired*。

## 4. 為什麼這樣能「超越」

1. **可驗證性**：FermiQLLM 0 篇論文、0 權重、0 第三方復現；我們每個數字都有指令與 JSON。
2. **同基座、同算力**：直接在 Qwen3 上對 dense 基線比，不用「同參數量級」模糊比較。
3. **負面結果公開**：舊 MPO 排序的失敗、SQA 不一定贏 greedy——都寫進報告，
   這是內容農場式宣傳無法複製的可信度。
4. **全鏈路真的閉環**：壓縮 → 退火分配 → 蒸餾 heal → 稀疏注意力 → 公開評測，
   一條指令 (`scripts/runpod_all_phases.sh`) 在單卡上跑完。

## 5. 時程與成本

| 階段 | 硬體 | 估計成本 | 時程 |
|---|---|---|---|
| QL1.5–QL4（本 PR，0.6B） | 1× H100 | ~US$10 | 已完成 |
| QL5（4B/8B） | 4× H100 | ~US$300–600 | 2–3 週 |
| QL6（Triton + vLLM） | 1× H100 | ~US$50 | 2 週 |
| QL7（RL + SQA） | 4× H100 | ~US$500 | 3–4 週 |
| QL8 | CPU / 雲 QPU | 小 | 持續 |
