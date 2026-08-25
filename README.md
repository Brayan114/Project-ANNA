# ANNA: Autonomous Neuromorphic Navigation Architecture

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![SystemVerilog](https://img.shields.io/badge/RTL-IEEE%201800--2017-orange.svg)]()
[![ASIC Process](https://img.shields.io/badge/ASIC-SkyWater%20130nm-green.svg)]()
[![Energy](https://img.shields.io/badge/Energy-0.42%20pJ%2FSynOp-brightgreen.svg)]()
[![Active Power](https://img.shields.io/badge/Power-%3C5%20%CE%BCW-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/Tests-13%2F13%20Passing-success.svg)]()

**A Microwatt-Scale Spiking Brain and Synthesizable ASIC Architecture for Autonomous Vector Path Integration and Landmark Navigation**

[**Read the Paper (PDF)**](docs/ant_neuromorphic_research_paper.pdf) • [**Silicon RTL**](hw/rtl/) • [**Benchmarks**](experiments/) • [**Documentation**](docs/)

</div>

---

## 🐜 Overview

**ANNA** (**A**utonomous **N**euromorphic **N**avigation **A**rchitecture) is an end-to-end, biologically grounded, and hardware-synthesizable neuromorphic navigation pipeline inspired by the celestial orientation and visual snapshot memory of the desert ant (*Cataglyphis fortis*).

While conventional robotic SLAM and deep neural networks require watts of power on GPUs, ANNA executes continuous closed-loop 2D path integration and visual landmark recovery on a **sub-$5\ \mu\text{W}$ digital silicon architecture** consuming **$0.42\text{ pJ per Synaptic Operation}$** in the open-source **SkyWater 130nm CMOS** node.

```
+--------------------------------------------------------------------------------------------------+
|                                    ANNA SYSTEM ARCHITECTURE                                      |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +--------------------------+       +-------------------------------+                           |
|   | Celestial Compass Sensor | ----> | PB Ring Attractor Compass     | (16 Glomerular Columns)   |
|   | (Polarized DRA E-vector) |       +-------------------------------+                           |
|   +--------------------------+                      |                                            |
|                                                     v                                            |
|   +--------------------------+       +-------------------------------+                           |
|   | Optic Flow Velocity      | ----> | FB CPU4 Path Integrator       | (Cosine Projected Memory) |
|   | (Translational Sensor)   |       +-------------------------------+                           |
|   +--------------------------+                      |                                            |
|                                                     v                                            |
|   +--------------------------+       +-------------------------------+                           |
|   | 360 Panoramic Retina     | ----> | MB Kenyon Cells (k-WTA < 5%)  | (1000 Sparse Neurons)     |
|   | (Compound Eye Skyline)   |       +-------------------------------+                           |
|   +--------------------------+                      |                                            |
|                                                     v                                            |
|                                      +-------------------------------+                           |
|                                      | 3-Factor R-STDP Dopamine LTD  | (Valleys of Familiarity)  |
|                                      +-------------------------------+                           |
|                                                     |                                            |
|                                                     v                                            |
|                                      +-------------------------------+                           |
|                                      | Pontine Steering Comparator   | (Closed-Loop Kinematics)  |
|                                      +-------------------------------+                           |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

---

## ⚡ Key Highlights

* **🧠 Biophysical Central Complex (CX):** 16-column Protocerebral Bridge (PB) continuous ring attractor tracking heading with zero drift, coupled to Fan-Shaped Body (FB) CPU4 velocity accumulators for dead-reckoning.
* **👁️ Mushroom Body (MB) Landmark Memory:** 1,000-neuron Kenyon Cell sparse expansion ($k$-WTA APL feedback maintaining $<5\%$ firing) with 3-Factor reward-modulated STDP (dopaminergic Anti-Hebbian LTD) for single-shot skyline learning.
* **🔁 Dual-Pathway Recovery:** Resolves catastrophic forced displacement ($6.40\text{ m}$) where pure dead-reckoning fails completely, recovering the nest with a **$67.73\%$ error reduction** ($p < 10^{-15}$).
* **💻 Synthesizable SystemVerilog ASIC (IEEE 1800-2017):** Fully synthesizable digital hardware core with 25-bit Address-Event Representation (AER) packet router, 16-bit Q4.12 fixed-point integer arithmetic, and multiplierless bit-shift exponential leak logic.
* **🔬 Silicon Post-Synthesis Metrics (SkyWater 130nm):**
  * **Core Die Area:** $0.383\text{ mm}^2$ (21,920 logic gates)
  * **Energy per SynOp ($E_{\text{SOP}}$):** **$0.42\text{ pJ / SynOp}$** ($>57,000\times$ lower energy than dense LSTM baselines)
  * **Active Power Envelope:** **$4.87\ \mu\text{W}$** at $1.2\text{V}$, $50\text{ MHz}$ (sub-$5\ \mu\text{W}$)

---

## 📊 Empirical Benchmarks

### 1. Statistical Monte Carlo Homing ($N=50$ Trials)
| Metric | Single Run | Monte Carlo Aggregate ($N=50$) | Ideal Target |
| :--- | :--- | :--- | :--- |
| **Final Homing Error ($\epsilon_{\text{home}}$)** | **$0.137\text{ m}$** | **$0.255 \pm 0.048\text{ m}$** | $0.000\text{ m}$ |
| **Path Tortuosity Index ($\tau$)** | **$0.999 \approx 1.000$** | **$1.016 \pm 0.008$** | $1.000$ (Straight-line) |
| **Neural Population Sparsity** | **$99.66\%$** | **$99.93\% \pm 0.01\%$** | $>95\%$ (Event-driven) |
| **Energy Reduction vs. LSTM** | **$99.98\%$** | **$99.98\%$** | $>90\%$ savings |

### 2. Synthesized Silicon Characteristics (SkyWater 130nm)
| Silicon Parameter | Synthesized Prototype Macro | Projected Full-Scale ASIC |
| :--- | :--- | :--- |
| **Neural Capacity** | 64 Vectorized LIF Neurons | 1,000 Kenyon Cells (Time-Multiplexed) |
| **Synaptic Capacity** | 1,024 Weights (16-bit Q4.12) | 36 KB On-Chip SRAM Crossbar |
| **Gate Count** | 21,920 NAND2 Equivalents | 28,400 NAND2 Equivalents |
| **Die Area** | $0.383\text{ mm}^2$ ($383\ \mu\text{m} \times 383\ \mu\text{m}$) | $0.850\text{ mm}^2$ ($922\ \mu\text{m} \times 922\ \mu\text{m}$) |
| **Operating Frequency** | 50.0 MHz (20 ns cycle time) | 50.0 MHz (20 ns cycle time) |
| **Energy per SynOp ($E_{\text{SOP}}$)** | **$0.42\text{ pJ / SynOp}$** | **$0.45\text{ pJ / SynOp}$** |
| **Total Active Power** | **$4.87\ \mu\text{W}$** | **$18.50\ \mu\text{W}$** |

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Brayan114/Project-ANNA.git
cd ANNA
pip install -r requirements.txt
```

### 2. Run Interactive Quickstart Demo
Run the end-to-end interactive simulation of ant foraging, celestial heading tracking, and visual landmark recovery:
```bash
python quickstart.py
```

### 3. Run Experiments & Generate Figures
```bash
# Experiment 1: Central Complex Path Integration & Homing
python experiments/exp1_path_integration.py

# Experiment 2: Forced Displacement Recovery via Mushroom Body
python experiments/exp2_visual_navigation.py

# Experiment 3: Silicon Synthesis & Energy Profiling
python experiments/exp3_chip_profiling.py

# Experiment 4: Statistical Monte Carlo Benchmark & Ablations (N=50 Trials)
python experiments/exp4_statistical_ablations.py
```

### 4. Run Automated Test Suite
Verify mathematical and hardware bit-exact parity:
```bash
pytest -v
```

---

## 📁 Repository Structure

```
ANNA/
├── docs/                                # Academic paper, LaTeX, and compiled PDF
│   ├── ant_neuromorphic_research_paper.pdf # Complete publication PDF (2.70 MB)
│   ├── academic_paper_manuscript.md     # Full Markdown manuscript
│   ├── manuscript.tex                   # LaTeX source
│   ├── references.bib                   # BibTeX citations
│   └── paper.typ                        # Typst source
├── figures/                             # 300 DPI publication-grade figures
│   ├── exp1_path_integration.png
│   ├── exp2_visual_navigation.png
│   ├── exp3_chip_microarchitecture.png
│   └── exp4_statistical_ablations.png
├── hw/                                  # Synthesizable SystemVerilog ASIC & Emulator
│   ├── rtl/                             # IEEE 1800-2017 synthesizable hardware
│   │   ├── defines.svh                  # Q4.12 fixed-point types & AER packet struct
│   │   ├── aer_router.sv                # 25-bit AER router with 16-deep circular FIFO
│   │   ├── lif_neuron_core.sv           # 16-channel vectorized LIF NPU core
│   │   ├── sram_synapse_matrix.sv       # Dual-port SRAM crossbar & 3-factor R-STDP ALU
│   │   └── ant_neuromorphic_top.sv      # Top-level SoC integration
│   └── sim/                             # Bit-exact cycle-accurate hardware emulator
│       └── chip_emulator.py
├── src/                                 # Python biophysical simulation engine
│   ├── central_complex/                 # PB ring attractor & FB velocity integrator
│   ├── mushroom_body/                   # Kenyon cells & 3-factor dopamine LTD
│   ├── dual_pathway_agent.py            # Unified CX+MB arbitration agent
│   └── lif_neuron.py                    # Vectorized LIF biophysical dynamics
├── env/                                 # 2D continuous desert arena & panoramic vision
│   ├── arena.py
│   ├── vision.py
│   └── sensors.py
├── experiments/                         # Reproducible benchmark experiments
├── tests/                               # Automated unit & hardware parity test suite
├── quickstart.py                        # Interactive 1-click demonstration script
├── setup.py                             # Python package definition
└── LICENSE                              # MIT License
```

---

## 📖 Citation

If you use ANNA or find this architecture helpful in your research, please cite:

```bibtex
@article{anna2026neuromorphic,
  title={Ant-Inspired Neuromorphic Computing: A Microwatt-Scale Spiking Brain and Synthesizable ASIC Architecture for Autonomous Vector Path Integration and Landmark Navigation},
  author={ANNA Research and Development Group},
  journal={arXiv preprint arXiv:2608.xxxxx},
  year={2026},
  url={https://github.com/Brayan114/Project-ANNA}
}
```

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
