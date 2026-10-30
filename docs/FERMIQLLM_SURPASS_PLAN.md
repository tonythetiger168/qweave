# 超越 FermiQLLM 1.0 — 改進計畫 (Surpass Plan)

> 2026-10-28 · 依據：[QL_RESULTS.md](QL_RESULTS.md)（H100 實測）、
> [FERMIQLLM_GLOBAL_INVESTIGATION.md](FERMIQLLM_GLOBAL_INVESTIGATION.md)、
> [QUANTUM_LLM_ROADMAP.md](QUANTUM_LLM_ROADMAP.md)。

## 0. 一句話

FermiQLLM 1.0 的五個「量子增強」環節只有公司內測數字。qweave 已把同一套技術
（張量網絡壓縮、量子模擬退火、糾纏啟發的結構）在 **Qwen3 同家族基座**上
**實測、公開、含負面結果**。下一步是用這些實測教訓，在**公開榜單**上以同等算力超過它。

## 1. 實測後我們知道的事（2026-10-28，1× H100，約 US$25）

| # | 發現 | 數據 | 對 FermiQLLM 主張的意義 |
|---|---|---|---|
| F1 | 原始範數下的 TN/MPO 壓縮不能用於已訓練權重 | 舊 MPO：PPL 21.7 → 755 084；TT-matrix：→ 747 971，訓練慢 34× | 「張量網絡嵌入模型結構」若不做激活感知，是 PR 不是工程 |
| F2 | 激活白化的 2-site MPO（SVD-LLM 式）才可行 | −30% 候選參數：PPL 199 → heal 後恢復 97–99% 差距 | 真正的技術核心是**激活感知低秩 + 蒸餾修復** |
| F3 | 領域內 heal 會讓壓縮模型「看起來比原模型好」 | WikiText heal：17.0 < dense 21.7；但 dense 同資料微調 150 步即 13.7 | 「比同參數模型強 +10–20%」若沒對**同配方微調的 dense**比較，可能只是領域適應 |
| F4 | PPL 掩蓋推理能力崩潰 | −11% 參數：PPL +16%，常識題持平，**MMLU −13pt、GSM8K −34pt** | MATH-500 / GPQA 宣稱必須公開逐題結果；小模型壓縮最先傷的就是數學與知識 |
| F5 | 「量子退火」對可分離的秩分配是錯工具 | greedy 距下界 0.6–0.9%（0.02 s）；SA 6–8%；SQA 19–32%（45 s） | 退火只在**有層間交互**的目標上才可能有價值，需實證 |
| F6 | 低秩層只有在秩對齊 tensor core 時才更快 | r=384：1.14× dense；r=477：0.39× | 「推理 +15%」取決於 kernel 細節，不是物理 |
| F7 | 量子啟發稀疏注意力在長上下文確實有效 | 16K：1.49×、PPL +0.96%；32K：3.47× | 這是 FermiQLLM **沒提**、而我們已過驗收門檻的方向 |
| F8 | 「糾纏熵決定稀疏預算」是負面結果 | 自適應 +6.9% vs 均勻 +3.8%（同平均 k） | 物理隱喻需實測，不能當設計原則直接用 |

## 2. 逐環節：FermiQLLM 主張 → qweave 現況 → 超越目標

| 環節 | FermiQLLM 1.0 | qweave 現況（已實測） | 超越目標（可驗證） |
|---|---|---|---|
| 模型結構 | TN 嵌入，推理 +15% | wsvd 2-site MPO，秩對齊；TT/MPO 負面結果公開 | QL5/QL6：−25% 參數、MMLU −≤1pt、decode ≥1.3× |
| 量子退火 | 用於優化 | SQA/SA/greedy/DP 全比較，結論：可分離問題用 DP/greedy | QL7：只在有交互項的目標上用，並與 greedy 對打 |
| 訓練/RL | RL 成本 −25% | 蒸餾 heal，TN-only heal VRAM −29% | QL7：GRPO 同分數下 GPU-hours ≤0.75× |
| 注意力 | 未提 | QSA 16K 1.49× @ +0.96% PPL（過門檻） | QL6：整合 vLLM prefill，32K+ ≥2× |
| 評測 | 內測 +10–20%，無權重 | lm-eval + 魯棒性套件，JSON 全公開 | QL8：公開榜單 + 權重 + 一鍵復現 |

## 3. 改進計畫（QL5–QL8），每項都由實測教訓推導

### QL5 — 保住知識與推理（修 F3/F4）
- **基座**：改用 `Qwen3-0.6B-Base` / `Qwen3-4B-Base` 做 PPL 參考（chat 模型微調會自降 PPL，見 F3）。
- **知識感知敏感度**：校準集混入 MMLU 式知識文本與 GSM8K 式推理鏈（不與評測重疊），
  秩分配目標從「激活誤差」改為「激活誤差 × 層的 Fisher 重要度」。
- **更長 heal**：≥100M tokens（目前 6M）、KD 溫度/比例掃描、對推理資料加權。
- **組合量化**：低秩 + INT8/FP8（小模型的 Pareto 前沿通常由量化主導，低秩要與之疊加才有意義）。
- **驗收**（out-of-domain）：−20% 參數時 MMLU −≤2pt、GSM8K −≤5pt；dense 同配方微調作對照。
- 成本：4× H100，約 US$300–600。
- **實作（2026-10-29）**：`phase4_llm/ql5_compress.py`（Fisher 加權秩分配、知識/推理校準混合 +
  13-gram 去污染、詞頻 × 隱狀態加權的嵌入/LM head 低秩、規範固定、INT8/FP8、分塊 KD 多 GPU heal），
  `ql5_gate.py`（全測試集 paired bootstrap 信賴區間）。指令與成本（實估 US$130–200）見
  [QL5_RUN_PLAN.md](QL5_RUN_PLAN.md)。尚無 GPU 實測數字。

### QL6 — 真的變快（修 F6，放大 F7）
- 秩一律對齊 64/128（已實作 `--rank-align`），融合 `x·Bᵀ·Aᵀ` 為單一 Triton kernel。
- batch=1 decode（記憶體頻寬受限）量測 tokens/s：這是低秩/TN 唯一天然佔優的場景。
- QSA 接入 vLLM 長上下文 prefill；用 top-16 預設（16K +0.96% PPL）。
- **驗收**：decode ≥1.3× dense（對 FermiQLLM「推理 +15%」），32K prefill ≥2×。
- **實作（2026-10-29）**：`phase4_llm/ql6_decode.py`。先用 batch-1 位元組屋頂線算上限：QL5 的 −20%
  在 0.6B / 4B 上限只有 ~1.2×（≈ FermiQLLM 的 +15%），**單靠低秩過不了 1.3×**；需要約 −30% 以上參數，
  或 **L2 無損投機解碼**（dense 驗證壓縮 draft 的 γ 個 token，輸出等同 dense greedy）。另含秩對齊、
  gate/up 兄弟融合、CUDA graph 引擎、torchao INT8（對照 dense INT8）、QSA 端到端 32K prefill。
  原計畫的單一 Triton kernel 改由 CUDA graph 取代（理由見 [QL6_RUN_PLAN.md](QL6_RUN_PLAN.md)）；
  vLLM 只作 dense 參考。
- **實測（2026-10-29/30，1× H100，S/L/K 三階段約 US$30）：未過門檻，記為負面結果。** batch-1 decode
  受 kernel 啟動數限制而非頻寬：低秩 0.75–0.95×（Triton kernel 約 0.5×）、INT8 ≤1×、投機解碼 0.6–0.8×
  （stage S 的 1.21× 為量測假象，已修）。32K prefill 端到端 1.52×（需 2×）。引擎差異最大：dense vLLM
  比 HF+CUDA graph 快 1.36–1.8×，所以任何「推理 +X%」都必須註明引擎。詳見 [QL6_RUN_PLAN.md](QL6_RUN_PLAN.md)。

### QL7 — 把「量子退火」放到對的問題上（修 F5/F8）
- 先量測層間交互：Hessian 交叉項 / 成對壓縮的非可加 PPL 變化；只有交互顯著時才用 SQA。
- RL：GRPO 中「本步更新哪些層/秩方向」寫成帶交互項的 QUBO，SQA vs greedy vs 全量 AdamW 對打。
- **驗收**：同最終分數下 GPU-hours ≤0.75×，否則公布為負面結果。

### QL8 — 公開、可復現、上榜
- 權重、配置、log 全上 HF；`scripts/runpod_all_phases.sh` 一條指令重跑。
- 提交 Open LLM Leaderboard 類公開評測；附 FermiQLLM 公開資料的逐項對照表。
- 量子硬體接口：Phase II 的 DEM + PyMatching 管線（已修復）保留給未來 QPU 子程序實驗；
  在出現可證明量子優勢前，一律稱 *quantum-inspired*。

## 4. 為什麼這樣能「超越」

1. **可驗證**：FermiQLLM 0 論文、0 權重、0 第三方復現。我們每個數字都有指令與 JSON。
2. **同基座、同配方對照**：F3 證明「比同參數模型強」很容易是領域適應造成的錯覺，我們把對照組做對。
3. **公布負面結果**：F1、F5、F8 正是「量子」敘事最容易誇大的地方，公開它們是可信度的護城河。
4. **已有一項過門檻的新方向**（F7, QSA），是 FermiQLLM 未主張的。

## 5. 時程與成本

| 階段 | 硬體 | 估計成本 | 時程 |
|---|---|---|---|
| QL1.5–QL4（本 PR，0.6B，已完成） | 1× H100 | ~US$25 | 完成 |
| QL5 知識保留 + 4B | 4× H100 | US$300–600 | 2–3 週 |
| QL6 kernel + vLLM | 1× H100 | ~US$50 | 2 週 |
| QL7 RL + 交互式 QUBO | 4× H100 | ~US$500 | 3–4 週 |
| QL8 公開榜單 | — | 小 | 持續 |
