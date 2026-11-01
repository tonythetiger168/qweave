# 全球量子物理研究前沿 (2025–2026) 與 qweave 改進計畫

> 2026-10-30 · 接續 [SURVEY.md](SURVEY.md)（2026-09，模擬器工程面）。本文改看**物理與演算法前沿**：
> 量子糾錯、解碼器、魔術態、古典模擬邊界（量子優勢之爭）、近期演算法、量子啟發 AI，
> 逐項對照 qweave 現況並提出可驗收的改進項目 QP1–QP7。
>
> **資料可靠度**：本文依據公開報導與論文摘要（網路檢索）。標 ⚠️ 者為 arXiv 預印本或二手來源，
> 數字尚未經同儕審查或未能直接核對原文（本 session 網路政策封鎖 arxiv.org）。

## 0. 摘要

2025–2026 年量子物理的主線已從「做出更多 qubit」轉為三件事：
**(1) 糾錯真的低於閾值並開始擴展**（表面碼、色碼、高碼率 qLDPC）；
**(2) 量子優勢主張與古典模擬（tensor network + belief propagation、Pauli propagation）的拉鋸**；
**(3) 演算法務實化**（取樣式對角化 SQD/SKQD 取代 VQE 成為近期化學預設；變分法「無 barren plateau ⇔ 可古典模擬」的疑慮）。

qweave 目前有 SV / Clifford / MPS / trajectory 模擬器、重複碼 DEM + PyMatching、
距離 3 表面碼、VQE/QAOA/QPE/IPEA。和前沿相比，最大缺口是：
**電路級雜訊下的可擴展 QEC 實驗（表面碼/色碼/qLDPC + 現代解碼器）**、
**Heisenberg 圖像的古典模擬器（Pauli propagation、BP 張量網路）**、
**取樣式對角化**。這三項也正好是 qweave「誠實驗證 quantum 主張」定位的核心工具。

## 1. 研究前沿盤點

### 1.1 量子糾錯：低於閾值、開始擴展

| 成果 | 數據 | 來源 |
|---|---|---|
| Google Willow 表面碼記憶 | d=7（101 qubits）每輪邏輯錯誤 0.143%，距離 +2 時抑制因子 Λ = 2.14 ± 0.02 | Nature 638 (2025)；2026-04 刊出更正 |
| 三角色碼低於閾值（Quantinuum Helios，離子阱） | d = 3, 5, 7 低於閾值 | ⚠️ arXiv:2610.24077（2026-10 預印本）|
| Quantinuum Helios | 98 個 Ba 離子，雙 qubit 保真度 99.921%；報導 48 個糾錯邏輯 qubit（94 為錯誤**偵測**）；2026-08 報「接近五個九」的邏輯保真度 | ⚠️ 二手彙整，數字各來源不一 |
| Harvard/MIT/QuEra 中性原子 | 448 原子上同時 96 個 d=4 邏輯 qubit，高碼率 [[16,6,4]] 區塊 | ⚠️ 二手來源，歸於 Nature 2026，未能核對 |
| IBM qLDPC 路線 | gross code（雙變數 bicycle 碼）宣稱糾錯開銷降約 90%；Loon (2025) 測試長程耦合、Kookaburra (2026)、Starling (2029, 200 邏輯 qubit、1 億 gate) | IBM 硬體頁 + 二手彙整 |
| union-find 解碼器有限閾值證明 | 理論結果 | ⚠️ arXiv:2602.20238 |

**趨勢**：表面碼 → 色碼（transversal Clifford）→ 高碼率 qLDPC（需長程連線，離子/中性原子/IBM 新耦合器）。
評估指標從單一「邏輯錯誤率」變成 **Λ（隨距離的抑制）+ 解碼延遲 + 每邏輯 qubit 的實體開銷**。

### 1.2 解碼器：從 MWPM 到即時 BP 家族

| 解碼器 | 重點 | 來源 |
|---|---|---|
| Relay-BP（IBM） | 串接多個 memory-BP；比 BP+OSD 準確約 10×；純訊息傳遞、可上 FPGA/ASIC | ⚠️ arXiv:2506.01779 等 |
| 全平行 BP | 去掉 Relay 的串列延遲，宣稱可勝 BP-OSD | ⚠️ arXiv:2507.00254 |
| FPGA gross-code 即時解碼 | 約 52% LUT；結論：訊息傳遞是大規模 qLDPC 即時解碼最實際的路線 | ⚠️ arXiv:2511.21660 |
| 相關錯誤解碼 | 電路級雜訊下處理 Y/相關錯誤；以 Tesseract（搜尋式）與 XYZ-Relay-BP（301 段 DMem-BP）為準確度參考 | Nature Communications (2026), s41467-026-70556-3 |

**趨勢**：MWPM 只適用圖狀（graph-like）錯誤；qLDPC 與相關錯誤需 BP 家族；**延遲**與準確度同等重要。

### 1.3 魔術態與資源估算

| 成果 | 數據 | 來源 |
|---|---|---|
| Magic state cultivation（Google） | 讓乾淨的 T/CCZ 態幾乎和一般操作一樣便宜；Willow 上實驗誤差降約 40× | 2024 理論；實驗見 Google（Science）⚠️ 日期不一 |
| RSA-2048 資源估算（Gidney） | < 100 萬含噪 qubit、< 1 週（2019 估計為 2000 萬 qubit、8 小時）；靠 cultivation + 近似剩餘算術 + yoked 表面碼 | arXiv:2505.15917 |
| 邏輯催化劑培養（dyadic 相位旋轉） | cultivation 的延伸 | ⚠️ arXiv:2606.27358 |

### 1.4 古典模擬邊界：量子優勢之爭

| 主張 | 古典回應 | 現況 |
|---|---|---|
| IBM Eagle kicked Ising (2023) | BP 張量網路兩週內複現（PRX Quantum 5, 010308, 2024）；MPS / BP-TN / Pauli propagation 皆可處理約 700 個雙 qubit gate 規模 | 普遍認為已被古典追上 |
| D-Wave 退火「超越古典」(Science 2025) | Flatiron + BU 以 3D BP 張量網路（ITensor）匹配精度 | D-Wave 反駁（biclique、最大 3D、部分 observable）；⚠️ 爭議中 |
| Google Quantum Echoes / OTOC（Nature, 2025-10）宣稱 13,000×、可驗證 | ⚠️ arXiv:2604.15427 主張 BP 張量網路**無法**可行地模擬此實驗 | 目前未見有效古典反駁；尚未由另一台量子電腦驗證 |
| 方法學 | Pauli propagation 框架（⚠️ arXiv:2505.21606）、實用指南（⚠️ 2507.10771）、強關聯系統（⚠️ 2511.21651）、對稱合併（⚠️ 2512.12094）；TN 觀點綜述（⚠️ 2603.18825） | 快速成長 |

**教訓**：每個優勢主張都需要「當時最好的古典演算法」作對照，而這些演算法（BP-TN、Pauli propagation）
已經是標準工具。這和我們在 QL 系列對 FermiQLLM 的結論相同：**主張必須附可復現的對照組**。

### 1.5 近期演算法

| 方向 | 重點 | 來源 |
|---|---|---|
| SQD / QSCI（取樣式量子對角化） | 綜述稱已成為容錯前化學的預設做法 | ⚠️ arXiv:2608.05314 |
| SKQD（取樣式 Krylov） | 取代 QPE 的容錯前方案；最強實驗為 49 qubit，但限於稀疏基態局域 Hamiltonian | arXiv:2501.09702；⚠️ 2608.05314 |
| SqDRIFT | 隨機化 SKQD，保留可證收斂（含隨機編譯與有限取樣誤差） | ⚠️ arXiv:2508.02578 |
| Schwinger model SKQD | 離子阱 + 超導兩平台上的格點規範理論 | ⚠️ arXiv:2510.26951 |
| Barren plateau ⇔ 可古典模擬 | 許多「可證無 barren plateau」的參數化電路，在初始資料收集後可古典模擬損失函數 | Cerezo 等, Nature Communications 16, 7907 (2025) |

### 1.6 量子啟發 AI

- CompactifAI（MPO 壓縮 LLM）：宣稱 −70% 參數、>90% 準確度（重訓後），結合量化達 88–93% 記憶體縮減；
  Multiverse 2026 開源 HyperNova 60B。⚠️ 多為公司自報。
- 我們自己的 QL1.5–QL4 實測（`docs/QL_RESULTS.md`，見 PR https://github.com/tonythetiger168/qweave/pull/1 ）：原始範數的 TN/MPO 對已訓練權重不可用；
  激活白化後可行，但小模型的知識/數學能力最先崩潰；「量子退火」在可分離分配問題上輸給 greedy。
  這與 1.4 節的教訓一致：**量子啟發不等於量子優勢，對照組決定一切**。

## 2. qweave 差距分析

| 面向 | qweave 現況 | 前沿 | 差距 |
|---|---|---|---|
| QEC 實驗 | 重複碼 DEM + PyMatching（已修復）；d=3 表面碼 tableau | 電路級雜訊表面碼 d=3–7 的 Λ、色碼、gross code | 無可擴展的旋轉表面碼/色碼/qLDPC 電路產生器；無 Λ 量測 |
| 解碼 | PyMatching (MWPM) | BP+OSD、Relay-BP、相關錯誤解碼、神經解碼 | 無 BP 家族；無延遲量測 |
| 雜訊模型 | 去極化、振幅/相位阻尼、trajectory | SI1000（超導）、leakage、erasure（中性原子/雙軌）、離子阱全連接 | 無硬體化雜訊模型與 leakage/erasure |
| 非 Clifford | SV（約 20–30 qubit）、MPS | Clifford + 少量 T（stabilizer rank）、cultivation 模擬 | 無「近 Clifford」模擬器 |
| 古典模擬邊界 | Schrödinger 圖像（SV/MPS） | Heisenberg 圖像 Pauli propagation、BP 張量網路 | 無法審核 utility 級實驗 |
| 演算法 | VQE、QAOA、QPE、IPEA、sparse VQE | SQD/SKQD、穩健相位估計 | 無取樣式對角化 |
| 資源估算 | 無 | 表面碼 / cultivation 成本模型 | 無 |

## 3. 改進計畫 QP1–QP7

每項都給出**可驗收**的標準，且盡量以已知公開數字（Stim、論文）作外部對照。

### QP1 — 電路級 QEC 實驗工廠（P0，2–3 週，CPU）
- 旋轉表面碼記憶實驗產生器（d = 3, 5, 7, 9；X/Z 基底；R 輪），色碼（三角 4.8.8 / 6.6.6），
  gross code [[144,12,12]] 記憶電路；全部輸出成 qweave Circuit IR 與 Stim 電路。
- 雜訊：SI1000、均勻去極化、加上測量/重置錯誤。
- 指標：每輪邏輯錯誤率、**Λ(d→d+2)**、閾值曲線。
- **驗收**：qweave 自建電路的 DEM 與 `stim.Circuit.generated` 同參數電路的 DEM 等價；
  表面碼在 SI1000、p≈0.1–0.2% 下 Λ 落在 2 附近並與 Stim+PyMatching 一致（誤差棒內）。

### QP2 — 解碼器套件：BP 家族 + 延遲（P0，3–4 週，GPU 可選）
- BP+OSD（對照基線）、Mem-BP / **Relay-BP** 自行實作（numpy 參考版 + torch 批次版，沿用 qweave GPU 後端）。
- 相關錯誤（Y = X·Z）處理：XYZ 型解碼或相關性重新加權。
- 量測每 shot 解碼延遲（CPU/GPU），與準確度並列報告。
- **驗收**：gross code 上 Relay-BP 邏輯錯誤率優於 BP+OSD（文獻宣稱約 10×，我們報告實測值）；
  表面碼上與 PyMatching 對照。

### QP3 — 硬體化雜訊模型（P1，2 週）
- leakage（含 leakage reduction unit）、erasure（中性原子原子損失、雙軌 qubit）、相關串擾。
- 平台預設：超導（SI1000）、離子阱（全連接、較慢）、中性原子（erasure 偏置、搬運錯誤）。
- 對接 QP1：同一個碼在不同平台的 Λ 與每邏輯 qubit 實體開銷比較表。
- **驗收**：erasure 轉換後閾值提升的定性結果與文獻一致；leakage 開關前後的 Λ 差異可重現。

### QP4 — Heisenberg 圖像古典模擬器（P0，3–4 週，CPU）
- **Pauli propagation**：觀測量反向傳播，依 Pauli 權重/係數截斷，支援雜訊（對 Pauli 項的阻尼）。
- **BP 張量網路**：PEPS / 圖狀 TN + belief propagation 規範（gauging）與期望值計算，heavy-hex 與 2D/3D 晶格。
- 「優勢審核」基準集：IBM 127-qubit kicked Ising、小規模 OTOC / Quantum Echoes 型電路、退火動力學。
- **驗收**：在可用 SV/MPS 精確驗證的小尺寸上誤差 < 1e-2；127-qubit kicked Ising 結果與已發表的古典/實驗值吻合；
  並如實記錄 OTOC 類電路在哪個深度開始失效（對照 ⚠️ arXiv:2604.15427 的主張）。

### QP5 — 近 Clifford 模擬與魔術態（P1，3 週）
- Clifford + 少量 T 模擬：stabilizer rank / sum-over-Cliffords（T 數 ≲ 40），補上 tableau 與 SV 之間的空檔。
- 小規模 magic state cultivation（d=3）電路的錯誤率模擬，與已發表的成本表比較。
- **驗收**：T 數 ≤ 20 的隨機 Clifford+T 電路與 SV 結果一致到 1e-10；cultivation 小規模結果的趨勢與論文相符。

### QP6 — 取樣式對角化取代 VQE 作化學預設（P1，2–3 週）
- SQD：取樣（含 trajectory 雜訊）→ 組態恢復（configuration recovery）→ 子空間投影 → scipy 稀疏對角化。
- SKQD：Krylov 時間演化電路（Trotter）取樣，沿用 IPEA/QPE 模組。
- 依 Cerezo 2025 的觀點，在文件中標示 qweave 的 VQE 範例屬「可能可古典模擬」類，改以 SQD 為主示範。
- **驗收**：H₂、LiH、N₂（小活性空間）基態能量與 FCI 差 < 1.6 mHa（化學精度）；
  與同 shot 數的 VQE 比較誤差與成本。

### QP7 — 資源估算器（P2，2 週）
- 輸入：演算法（QPE 化學、Shor）、碼、實體錯誤率、cycle 時間 → 實體 qubit 數、時間、魔術態工廠佔比。
- 內建 cultivation / 傳統蒸餾成本模型、yoked 表面碼選項。
- **驗收**：在相同假設下複現 Gidney 2025 RSA-2048 估算的量級（< 100 萬 qubit、< 1 週）。

## 4. 優先順序與時程

| 優先 | 項目 | 理由 | 時程 | 算力 |
|---|---|---|---|---|
| P0 | QP1 QEC 實驗工廠 | 所有 QEC 研究的基礎；可用 Stim 嚴格對照 | 第 1–3 週 | CPU |
| P0 | QP4 Pauli propagation + BP-TN | 「審核量子主張」的核心工具，與 qweave 定位一致 | 第 1–4 週（並行） | CPU |
| P0 | QP2 BP 家族解碼器 | qLDPC 與相關錯誤必需；GPU 批次版可用現有後端 | 第 3–7 週 | CPU / 1 GPU |
| P1 | QP6 SQD/SKQD | 近期化學的實際預設，取代 VQE 示範 | 第 5–8 週 | CPU |
| P1 | QP3 硬體化雜訊 | 讓 QP1/QP2 的結論可對應真實平台 | 第 6–8 週 | CPU |
| P1 | QP5 近 Clifford / cultivation | 補足非 Clifford 空檔 | 第 8–11 週 | CPU |
| P2 | QP7 資源估算器 | 串起所有模組，對外最易理解 | 第 11–12 週 | CPU |

全部以 CPU 為主；GPU 只用於 QP2 批次解碼與 QP4 大型 BP 收縮，估計 RunPod 成本 < US$50
（沿用 `runpod_all_phases.sh` 的「先上傳、後關機」流程）。

## 5. 與 LLM 路線（QL 系列）的交集

- **同一個 BP 引擎**：QP4 的 belief propagation 張量網路可直接用於 QL7 的「有層間交互的分配目標」，
  取代在可分離問題上已證明無效的 SQA。
- **神經解碼器**（可選）：以 QL 系列的 GPU 訓練基礎設施，訓練小型 transformer 解碼器（AlphaQubit 型），
  與 QP2 的 BP 家族在同一 DEM 上對打——這是「AI for quantum」少數有明確對照組的方向。
- **方法論一致**：每個量子主張（硬體優勢或「量子增強」AI）都要附上最強的古典對照；qweave 提供這套對照工具。

## 6. 來源

- Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature 638 (2025) — https://www.nature.com/articles/s41586-024-08449-y （arXiv:2408.13687）
- ⚠️ *Quantum error correction well below the color code threshold* (Helios), arXiv:2610.24077 — https://arxiv.org/abs/2610.24077
- ⚠️ *Proof of a finite threshold for the union-find decoder*, arXiv:2602.20238 — https://arxiv.org/pdf/2602.20238
- ⚠️ *Improved belief propagation is sufficient for real-time decoding…* (Relay-BP), arXiv:2506.01779 — https://arxiv.org/pdf/2506.01779
- ⚠️ *Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD*, arXiv:2507.00254 — https://arxiv.org/pdf/2507.00254
- ⚠️ *FPGA-tailored algorithms for real-time decoding of quantum LDPC codes*, arXiv:2511.21660 — https://arxiv.org/html/2511.21660
- *Decoding correlated errors in quantum LDPC codes*, Nature Communications (2026) — https://www.nature.com/articles/s41467-026-70556-3
- IBM Relay-BP 報導 — https://quantumcomputingreport.com/ibm-researchers-devise-relay-bp-algorithm-for-real-time-decoding-of-qldpc-codes/
- Gidney, *How to factor 2048 bit RSA integers with less than a million noisy qubits*, arXiv:2505.15917 — https://arxiv.org/pdf/2505.15917
- Magic state cultivation on Willow（解說）— https://postquantum.com/quantum-research/google-magic-state-cultivation/
- ⚠️ *Cultivating logical catalysts for fault-tolerant dyadic phase rotations*, arXiv:2606.27358 — https://arxiv.org/pdf/2606.27358
- Google, *A verifiable quantum advantage* (Quantum Echoes) — https://research.google/blog/a-verifiable-quantum-advantage/
- Science News, *Quantum 'echoes'…* — https://www.sciencenews.org/article/quantum-echoes-google-computer
- ⚠️ *Tensor Networks with Belief Propagation Cannot Feasibly Simulate Google's Quantum Echoes Experiment*, arXiv:2604.15427 — https://arxiv.org/pdf/2604.15427
- ⚠️ *Quantum Advantage: a Tensor Network Perspective*, arXiv:2603.18825 — https://arxiv.org/pdf/2603.18825
- *A brief history of quantum vs classical computational advantage*, arXiv:2412.14703 — https://arxiv.org/html/2412.14703v2
- ⚠️ Pauli propagation：arXiv:2505.21606、2507.10771、2511.21651、2512.12094 — https://arxiv.org/pdf/2505.21606 · https://arxiv.org/pdf/2507.10771 · https://arxiv.org/pdf/2511.21651 · https://arxiv.org/pdf/2512.12094
- D-Wave vs Flatiron 爭議 — https://quantumcomputingreport.com/flatiron-institute-tensor-network-algorithm-overturns-historical-d-wave-quantum-supremacy-claim/ · https://quantumcomputingreport.com/d-wave-systematically-rebuts-flatiron-claims-reaffirming-beyond-classical-simulation-milestones/
- SKQD, arXiv:2501.09702 — https://arxiv.org/pdf/2501.09702 ；⚠️ SqDRIFT, arXiv:2508.02578 — https://arxiv.org/abs/2508.02578 ；⚠️ Schwinger SKQD, arXiv:2510.26951 — https://arxiv.org/pdf/2510.26951 ；⚠️ SQD 綜述, arXiv:2608.05314 — https://arxiv.org/html/2608.05314v2
- Cerezo 等, *Does provable absence of barren plateaus imply classical simulability?*, Nature Communications 16, 7907 (2025) — https://arxiv.org/abs/2312.09121
- IBM 量子硬體與路線圖 — https://www.ibm.com/quantum/hardware ；⚠️ 2026 路線圖彙整 — https://andesqubit.com/en/blog/quantum-hardware-roadmaps-2026-2030/ ；⚠️ Helios/QuEra 數字彙整 — https://quantumzeitgeist.com/what-is-quantum-error-correction/
- CompactifAI, arXiv:2401.14109 — https://arxiv.org/pdf/2401.14109v2 ；ESANN 2025 — https://www.esann.org/sites/default/files/proceedings/2025/ES2025-8.pdf
