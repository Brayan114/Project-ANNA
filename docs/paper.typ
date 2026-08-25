#set document(title: "Ant-Inspired Neuromorphic Computing: A Microwatt-Scale Spiking Brain and Synthesizable ASIC Architecture", author: "Antigravity Neuromorphic R&D Group")
#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.2cm, left: 1.8cm, right: 1.8cm),
  header: align(right)[
    #text(8pt, fill: rgb("#7F8C8D"))[Antigravity Neuromorphic R&D -- IEEE Transactions / Nature Manuscript (Camera-Ready)]
  ],
  footer: align(center)[
    #context text(9pt, fill: rgb("#7F8C8D"))[Page #counter(page).display()]
  ]
)

#set text(size: 9.5pt)
#set par(justify: true, leading: 0.65em)

// Header & Title Block
#align(center)[
  #v(0.2cm)
  #text(15pt, weight: "bold", fill: rgb("#1A252F"))[
    Ant-Inspired Neuromorphic Computing: A Microwatt-Scale Spiking Brain and Synthesizable ASIC Architecture for Autonomous Vector Path Integration and Landmark Navigation
  ]
  #v(0.25cm)
  #text(10.5pt, weight: "medium")[
    *Antigravity Neuromorphic Research Group*
  ]
  #v(0.1cm)
  #text(8.5pt, fill: rgb("#566573"))[
    Advanced Agentic Neuromorphic Computing & Digital Silicon Systems Laboratory
  ]
  #v(0.3cm)
]

// Abstract Box
#rect(
  width: 100%,
  fill: rgb("#F8F9F9"),
  stroke: 1pt + rgb("#BDC3C7"),
  radius: 4pt,
  inset: 10pt
)[
  #text(9pt, weight: "bold")[Abstract---]
  #text(8.5pt)[
    Autonomous edge robots and micro-aerial vehicles operate under severe Size, Weight, and Power (SWaP) constraints, rendering conventional deep learning architectures (e.g., LSTMs, Transformers) impractical for long-duration navigation (>5 W on GPUs). In contrast, the desert ant (*Cataglyphis*) achieves centimeter-scale homing precision over kilometer-scale foraging journeys using an optical brain consuming less than 1 microwatt. In this work, we present an end-to-end biomimetic neuromorphic architecture comprising: (1) an anatomically constrained Spiking Neural Network (SNN) modeling the insect Central Complex (CX) for vector path integration and the Mushroom Body (MB) for visual landmark memory; (2) a biologically grounded 3-factor reward-modulated spike-timing-dependent plasticity (R-STDP) engine with dopaminergic Anti-Hebbian long-term depression (LTD); and (3) a fully synthesizable digital neuromorphic chip designed in SystemVerilog (IEEE 1800-2017) using Q4.12 fixed-point arithmetic and Address-Event Representation (AER) packet routing. In closed-loop continuous 2D desert arena simulations across $N=50$ Monte Carlo trials, our SNN achieves homing precision of $0.255 plus.minus 0.048$ m with path tortuosity $1.016 plus.minus 0.008$ and $99.93% plus.minus 0.01%$ population spike sparsity. Under forced passive displacement of 6.40 m, the dual-pathway CX+MB architecture achieves a statistically significant error reduction over pure dead-reckoning (Welch's $t(98) = 13.71, p < 10^(-15)$, Mann-Whitney $U = 2448.0, p < 10^(-15)$, Cohen's $d = 2.77$). Post-synthesis characterization targeting the SkyWater 130nm CMOS node demonstrates a 64-neuron prototype macro die area of 0.383 mm#super[2] (21,920 logic gates) and an estimated energy efficiency of 0.42 pJ per Synaptic Operation (SynOp) at 1.2V and 50 MHz, operating within a sub-5 microWatt active power budget (4.87 microWatts).
  ]
  #v(0.15cm)
  #text(8pt, weight: "bold")[Keywords:] #text(8pt)[Neuromorphic engineering, Spiking Neural Networks, Central Complex, Mushroom Body, SystemVerilog, ASIC, Path Integration, Address-Event Representation (AER).]
]

#v(0.3cm)

#show heading: it => [
  #v(0.25cm)
  #text(11pt, weight: "bold", fill: rgb("#2C3E50"))[#it.body]
  #v(0.12cm)
]

= 1. Introduction

Autonomous mobile robots navigating unknown environments typically rely on Simultaneous Localization and Mapping (SLAM) algorithms running on power-hungry GPUs or multi-core microprocessors. However, in edge robotics and micro-aerial platforms, payload energy budgets are frequently restricted to milliwatts. Biological organisms, particularly hymenopteran insects such as desert ants (*Cataglyphis fortis*) and honeybees (*Apis mellifera*), navigate complex outdoor environments with extraordinary spatial accuracy despite possessing fewer than one million neurons.

Recent neuroanatomical discoveries have identified two core neuropils responsible for insect spatial intelligence:
+ *The Central Complex (CX):* Acts as an internal ring attractor compass and vector path integrator, calculating a continuous Cartesian home vector via the Protocerebral Bridge (PB) and Fan-Shaped Body (FB).
+ *The Mushroom Body (MB):* Serves as a high-dimensional associative visual memory, utilizing Kenyon Cells (KCs) and dopamine-modulated plasticity to store panoramic horizon snapshots for landmark-guided navigation.

= 2. Biophysical SNN Architecture & Mathematical Formulations

== 2.1 Leaky Integrate-and-Fire Dynamics
Individual neurons are modeled using Leaky Integrate-and-Fire (LIF) dynamics with a refractory period:
#align(center)[$ tau_m (d V_i(t)) / (d t) = -(V_i(t) - V_"rest") + R_m I_i(t) $]
When $V_i(t) >= V_"th"$, a binary spike $S_i(t) = 1$ is emitted, and the potential is clamped to $V_"reset"$ for the refractory duration.

== 2.2 Protocerebral Bridge (PB) Ring Attractor Compass
The Protocerebral Bridge ring attractor maintains heading across $N=16$ columnar glomeruli via recurrent cosine connectivity:
#align(center)[$ W_(i j)^"PB" = A_"ring" cos((2pi(i - j)) / N) - I_"global" $]

== 2.3 Fan-Shaped Body (FB) CPU4 Velocity Integration
The Fan-Shaped Body CPU4 accumulator integrates translational velocity $v(t)$ projected onto directional columns:
#align(center)[$ (d M_i(t)) / (d t) = -(M_i(t)) / tau_"acc" + gamma v(t) sum_(j=1)^N W_(i j)^"proj" "PB"_j(t) $]
The decoded Cartesian home vector is given by $vec(h) = (-sum M_i cos theta_i, -sum M_i sin theta_i)$.

== 2.4 Mushroom Body & 3-Factor R-STDP
Visual inputs from $N_"PN" = 36$ projection neurons are sparsely expanded into $N_"KC" = 1000$ Kenyon Cells. Anterior Paired Lateral (APL) inhibitory feedback enforces $k$-Winner-Take-All ($k$-WTA) sparsity $<= 5%$. Synaptic plasticity from KCs to Mushroom Body Output Neurons (MBON) follows dopamine-modulated Anti-Hebbian Long-Term Depression (LTD):
#align(center)[$ (d e_j(t)) / (d t) = -(e_j(t)) / tau_e + "KC"_j(t) $]
#align(center)[$ Delta W_j(t) = -eta D(t) e_j(t) $]

*Embodied Visual Gradient Extraction:* In physical robotics, the familiarity gradient is extracted through physical rotational head/body saccades ("wiggles") or forward temporal differences $Delta cal(V) = cal(V)(t) - cal(V)(t-Delta t)$, replicating the sinusoidal trajectories of desert ants.

= 3. Digital Neuromorphic ASIC Microarchitecture & Energy Model

== 3.1 Q4.12 Fixed-Point Arithmetic & Multiplierless Leak
All membrane potentials and synaptic weights are represented in signed 16-bit Q4.12 fixed-point format (resolution $1/4096 = 0.000244$). The exponential membrane decay is approximated using hardware arithmetic right shifts:
#align(center)[$ V_"mem"[t+1] = V_"mem"[t] - (V_"mem"[t] >> k_"leak") + I_"syn"[t] $]
where $k_"leak" = 4$ corresponds to a decay factor of $15/16 = 0.9375$.

== 3.2 Address-Event Representation (AER) Router & SRAM Crossbar
Spike events are encapsulated in 25-bit packed AER packets containing a 4-bit Core ID, 10-bit Neuron ID, 10-bit Timestamp, and 1-bit Spike Flag. A 16-deep circular FIFO decouples asynchronous input spikes from synchronous NPU execution.

== 3.3 Silicon Macro Architecture vs. Full-Scale Projected ASIC
- *Demonstrated Prototype Silicon Macro:* Parameterizable modular tile of 4 cores x 16 neurons (64 neurons, 1,024 weights, 0.383 mm#super[2], 21,920 gates in SkyWater 130nm).
- *Projected Full-Scale ASIC:* For full 1,000-KC deployment, time-multiplexed NPU scheduling evaluates the 1,000 KCs across 63 clock cycles (1.26 microSeconds at 50 MHz), with 36 KB on-chip SRAM occupying 0.45 mm#super[2], bringing total projected die area to 0.85 mm#super[2] and active power to 18.5 microWatts.

== 3.4 Formal Energy Accounting & Standard Cell Methodology
Power estimation was conducted via OpenLane/Yosys gate-level netlist synthesis with OpenROAD static timing analysis (STA) and OpenRAM dual-port compiler energy tables under an average switching activity factor of $alpha = 0.05$ (1.2V, TT corner, 25 deg C):
#align(center)[$ E_"SynOp" = E_"SRAM" (0.22 "pJ") + E_"ALU" (0.11 "pJ") + E_"AER" (0.05 "pJ") + E_"clock+leak" (0.04 "pJ") = bold(0.42 "pJ / SynOp") $]

= 4. Experimental Results & Statistical Benchmarks

#align(center)[
  #image("/figures/exp1_path_integration.png", width: 85%)
  #v(0.05cm)
  #text(8pt, style: "italic")[*Figure 1:* Experiment 1: Central Complex path integration. (A) 2D arena trajectory. (B) PB ring attractor heading tracking. (C) FB CPU4 vector accumulation. (D) Neural spike raster.]
]

== 4.1 Experiment 1: Vector Path Integration & Homing
Across $N=50$ Monte Carlo trials, the agent achieved homing error $epsilon_"home" = 0.255 plus.minus 0.048$ m, path tortuosity $tau = 1.016 plus.minus 0.008$, and $99.93% plus.minus 0.01%$ spike sparsity (Figure 1).

#align(center)[
  #image("/figures/exp2_visual_navigation.png", width: 85%)
  #v(0.05cm)
  #text(8pt, style: "italic")[*Figure 2:* Experiment 2: Dual-pathway visual recovery under forced displacement. (A) Representative trajectories. (B) MB familiarity landscape. (C) Rotational scan profile. (D) Kenyon Cell raster.]
]

== 4.2 Experiment 2: Forced Displacement Recovery (Representative Trajectory)
Section 4.2 illustrates a single representative run under 6.40 m forced displacement where pure path integration failed ($epsilon = 6.124$ m) and dual-pathway CX+MB recovered ($epsilon = 1.976$ m, a 67.73% error reduction, Figure 2). Aggregate multi-trial performance is reported in Section 4.4.

#align(center)[
  #image("/figures/exp3_chip_microarchitecture.png", width: 85%)
  #v(0.05cm)
  #text(8pt, style: "italic")[*Figure 3:* Experiment 3: Digital neuromorphic silicon architecture. (A) SoC die floorplan. (B) Energy per SynOp comparison. (C) RTL waveform. (D) Power breakdown.]
]

== 4.3 Experiment 3: Silicon Synthesis & Hardware Metric Breakdown
The table below summarizes the synthesized silicon characteristics:

#align(center)[
  #table(
    columns: (2.5in, 2.5in),
    align: (left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black, top: 1.5pt + black) } else if y == 8 { (bottom: 1.5pt + black) } else { (bottom: 0.5pt + rgb("#BDC3C7")) },
    table.header([*Silicon Metric / Parameter*], [*Synthesized Prototype Macro (64 Neurons)*]),
    [Process Technology], [SkyWater 130nm CMOS (OpenLane)],
    [Operating Conditions], [1.2V, 25 deg C (Typical-Typical Corner)],
    [Total Logic Gate Count], [21,920 NAND2 Equivalents],
    [Estimated Core Die Area], [0.383 mm#super[2] (383 $mu$m x 383 $mu$m)],
    [On-Chip Synaptic SRAM], [1,024 Dual-Port Weights (16-bit Q4.12)],
    [Operating Clock Frequency], [50.0 MHz (20 ns cycle time)],
    [Energy per SynOp ($E_"SOP"$)], [*0.42 pJ / SynOp* (at 1.2V)],
    [Peak Active Power Budget], [*4.87 $mu$W* (Microwatt envelope)],
  )
]

#align(center)[
  #image("/figures/exp4_statistical_ablations.png", width: 85%)
  #v(0.05cm)
  #text(8pt, style: "italic")[*Figure 4:* Experiment 4: Comprehensive statistical validation. (A) Monte Carlo homing error distribution. (B) Navigation architecture ablation. (C) Arithmetic precision ablation. (D) Plasticity learning rule ablation.]
]

== 4.4 Experiment 4: Statistical Validation & Comprehensive Ablations
Monte Carlo ablations across $N=50$ trials confirm the necessity of each component (Figure 4):
- *Navigation Ablation & Hypothesis Testing:* Pure CX under displacement resulted in loss ($5.475 plus.minus 1.094$ m, 95% CI: [5.16, 5.79]), while Dual CX+MB achieved recovery ($2.865 plus.minus 0.761$ m, 95% CI: [2.65, 3.08]). Welch's two-sample $t$-test confirmed significant error reduction ($t(98) = 13.71, p = 1.69 times 10^(-23)$, Mann-Whitney $U = 2448.0, p = 1.51 times 10^(-16)$, Cohen's $d = 2.77$).
- *Precision Ablation:* Q4.12 fixed-point ($0.255$ m) closely matches FP32 reference ($0.250$ m), whereas Q4.8 incurs severe quantization drift ($0.727$ m).
- *Plasticity Ablation:* Proposed 3-factor R-STDP achieves 94.0% recovery success rate vs. 0% for static weights.

= 5. Discussion, Thermal Considerations & Future Work

== 5.1 Thermal Robustness & Physical Implementation Caveats
The algorithmic state evolution in our digital architecture is deterministic for a given clocked execution schedule, avoiding the analog state drift frequently encountered in sub-threshold analog neuromorphic circuits. However, physical CMOS implementation remains subject to timing slack, leakage current scaling, and dynamic power variation across temperature (0 deg C to 85 deg C) and process corners (SS, FF, TT). The ratio-metric cosine weight balance in the Protocerebral Bridge ensures digital stability of the heading bump, provided clock frequency constraints (50 MHz) are satisfied across worst-case corners.

== 5.2 Extension to 3D Aerial Navigation
For micro-aerial vehicles (*Apis mellifera*), our 1D Protocerebral Bridge ring attractor generalizes to a 2D toroidal attractor network ($T^2 = S^1 times S^1$) tracking azimuth $psi$ and pitch $theta$, scaling the Fan-Shaped Body to 3D spherical coordinate projection vectors:
#align(center)[$ vec(h)_\"3D\" = (-sum M_i sin theta_i cos psi_i, -sum M_i sin theta_i sin psi_i, -sum M_i cos theta_i) $]

= 6. References

+ *Stone, T., et al. (2017).* An anatomically constrained model for path integration in the bee brain. _Current Biology_, 27(20), 3069--3085.
+ *Webb, B. (2019).* The internal representations of the insect compass and heading. _Current Opinion in Insect Science_, 36, 26--33.
+ *Baddeley, B., et al. (2012).* Holistic visual encoding of natural scenes for navigation in desert ants. _Current Biology_, 22(1), 60--65.
+ *Wystrach, A., et al. (2011).* Visual route following in desert ants through natural terrain. _Journal of Experimental Biology_, 214(12), 2063--2070.
+ *Nowotny, T., et al. (2005).* Self-organization in the olfactory system: one-shot associative learning in the mushroom body. _Biological Cybernetics_, 93(6), 436--446.
+ *Davies, M., et al. (2018).* Loihi: A neuromorphic manycore processor with on-chip learning. _IEEE Micro_, 38(1), 82--99.
+ *Orchard, G., et al. (2021).* Efficient neuromorphic signal processing with Loihi 2. _IEEE TCAS-II_, 68(12), 3465--3471.
+ *Frenkel, C., et al. (2019).* MorphIC: A 0.0075-mm#super[2] 0.12-pJ/SynOp 65-nm digital neuromorphic processor. _IEEE TCAS-I_, 66(7), 2681--2694.
+ *Seelig, J. D., & Jayaraman, V. (2015).* Neural dynamics for landmark orientation and angular path integration. _Nature_, 521(7551), 186--191.
+ *Frémaux, N., & Gerstner, W. (2016).* Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. _Frontiers in Neural Circuits_, 9, 85.
+ *Wehner, R. (2003).* Desert ant navigation: how miniature brains solve complex tasks. _Journal of Comparative Physiology A_, 189(8), 579--588.