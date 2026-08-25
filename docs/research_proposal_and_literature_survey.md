# Ant-Inspired Neuromorphic Computing: Bio-Realistic Spiking Architectures for Ultra-Low-Power Autonomous Navigation and Digital Silicon Realization

**Author / Candidate:** Doctoral Researcher  
**Domain:** Computational Neuroscience, Neuromorphic Engineering, Digital Microarchitecture, Bio-Inspired Robotics  
**Document Type:** Formal PhD Research Proposal & Comprehensive Literature Survey  
**Date:** August 2026  

---

## Executive Abstract

Autonomous agents operating at the extreme edge (e.g., micro-aerial vehicles, planetary rovers, subterranean exploratory probes) face severe compute, memory, and energy bottlenecks. Modern deep learning approaches relying on dense Artificial Neural Networks (ANNs) and GPU accelerators demand tens to hundreds of watts, precluding deployment on sub-watt swarms. 

In contrast, solitary foraging insects—particularly desert ants (*Cataglyphis fortis*) and honeybees (*Apis mellifera*)—navigate kilometers of complex, featureless terrain and return home along direct vectors using an active neural budget of fewer than **$10^6$ neurons** operating at **sub-milliwatt power budgets ($\approx 10^{-4}\text{ W}$)**.

This research project aims to:
1. **Reverse-engineer** and formulate the multi-modal neural microcircuits of the ant brain—specifically the **Central Complex (CX)** for celestial compass-heading integration and vector arithmetic, and the **Mushroom Body (MB)** for sparse visual-olfactory associative memory.
2. **Translate** these biological microcircuits into an event-driven, mathematically rigorous **Spiking Neural Network (SNN)** incorporating biologically grounded synaptic plasticity (three-factor reward-modulated STDP and homeostatic scaling).
3. **Design, simulate, and verify** a dedicated **digital neuromorphic hardware architecture (RTL)** with Address-Event Representation (AER) packet-based spike routing and event-driven fixed-point neuron processing units, targeting an open-source ASIC synthesis flow (OpenLane / SkyWater 130nm).
4. **Publish** seminal contributions in top-tier robotics, neuromorphic engineering, and computational neuroscience venues.

---

## 1. Problem Formulation & Academic Motivation

### 1.1 The Edge AI Compute & Energy Wall
Contemporary autonomous navigation pipelines depend heavily on simultaneous localization and mapping (SLAM), deep reinforcement learning (DRL), and dense visual odometry. These pipelines suffer from:
* **High Power Dissipation:** 10W–50W on modern edge accelerators (e.g., Nvidia Jetson Orin), incompatible with milligram-to-gram-scale robotic platforms.
* **Catastrophic Forgetting & Sample Inefficiency:** Inability to perform continuous, low-power, single-shot spatial memory updates without backpropagation over large replay buffers.
* **Synchronous Frame-Based Latency:** Wasted cycles processing static spatial frames rather than asynchronous, event-driven spatiotemporal changes.

### 1.2 The Biological Ant Paradigm (*Cataglyphis*)
The desert ant *Cataglyphis* navigates kilometers in Sahara desert heat, foraging in pseudo-random search paths before returning directly home via the shortest path (**path integration / dead reckoning**) using:
* **Polarized Light & Solar Compass:** Specialised dorsal rim area (DRA) ommatidia detecting celestial E-vector polarization angles.
* **Optic Flow & Step Counters:** Proprioceptive stride counting and ventral optic flow integration for ground velocity estimation.
* **Visual Snapshot Memory:** View-based matching to align with panoramic skyline panoramas near the nest.

These systems operate concurrently in specialized anatomical neuropils:
$$\text{CX (Vector Steering \& Orientation)} \longleftrightarrow \text{MB (Contextual Snapshot \& Odor Associative Memory)}$$

```
+---------------------------------------------------------------------------------------------------+
|                                  ANT SENSORY & NEURAL HIERARCHY                                   |
+---------------------------------------------------------------------------------------------------+
|  [Dorsal Rim Area (DRA)]   [Compound Eyes (Ventral Flow)]   [Antennae (Chemosensory / Olfactory)] |
|             |                             |                                       |               |
|             v                             v                                       v               |
|    [Anterior Optic Tubercle]    [Optic Glomeruli / Lobula]              [Antennal Lobe (AL)]      |
|             |                             |                                       |               |
|             +--------------+--------------+                                       |               |
|                            |                                                      |               |
|                            v                                                      v               |
|                  [ Central Complex (CX) ]                              [ Mushroom Body (MB) ]     |
|             - Protocerebral Bridge (Compass)                     - Kenyon Cells (Sparse Expansion)|
|             - Fan-Shaped Body (Path Integrator)                  - MBONs (Valence / Novelty)      |
|             - Noduli (Speed / Motion Gating)                     - Dopaminergic Plasticity        |
|                            \                                                     /                |
|                             \                                                   /                 |
|                              v                                                 v                  |
|                           [ Premotor Lateral Accessory Lobes (LAL) ]                              |
|                                                |                                                  |
|                                                v                                                  |
|                                 [ Motor Steering & Propulsion ]                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Comprehensive Literature Survey

### 2.1 Neurobiology of the Insect Navigation Circuit

#### 2.1.1 Central Complex (CX) Connectomics
The Central Complex is an evolutionarily conserved set of midline neuropils in arthropod brains consisting of:
1. **Protocerebral Bridge (PB):** Functions as a **ring attractor** maintaining a stable bump of neural activity representing current azimuthal heading ($\theta \in [0, 2\pi)$).
2. **Ellipsoid Body (EB) / Lower Division:** Connects ring neurons to compass neurons (EPG / columnar cells), transmitting polarized light and visual cue signals.
3. **Fan-Shaped Body (FB) / Upper Division:** Coordinates coordinate transformations and vector memory. CPU4 / PF-Col neurons accumulate optic flow velocity modulated by heading direction to maintain a continuous displacement vector $(\vec{x}, \vec{y})$.
4. **Noduli (NO):** Provides asymmetric self-motion signals and roll/pitch optic flow gating.

*Key Literature:*
* **Stone et al. (2017)** (*Current Biology*): Demonstrated a functionally complete, biologically verified neural circuit model of the bee/ant central complex executing path integration and steering on 8-column neural architectures.
* **Heinze et al. (2011–2018)** (*Nature Communications, J. Comp. Neurol.*): Detailed anatomical mapping of polarized light pathways from the locust/ant DRA to the CX.
* **Green et al. (2017) & Turner-Evans et al. (2020)** (*Nature, Neuron*): Identified ring-attractor dynamics and dynamic phase shifts in *Drosophila* CX EPG neurons.

#### 2.1.2 Mushroom Body (MB) and Contextual Associative Learning
The Mushroom Body is responsible for high-dimensional, sparse associative representations (visual snapshots and olfactory profiles):
1. **Projection Neurons (PN):** Transmit low-dimensional raw sensory cues from antennal and optic lobes.
2. **Kenyon Cells (KC):** Random high-dimensional projection ($\approx 2,000\text{--}100,000$ cells) with high firing thresholds producing extreme population sparsity ($< 5\%$ active simultaneously).
3. **Mushroom Body Output Neurons (MBON):** Read out KC activations to produce steering bias, valence (attractive vs. aversive), or familiar vs. novel ratings.
4. **Dopaminergic Neurons (DAN):** Encode reward/punishment signals triggering three-factor synaptic weight updates on KC-to-MBON synapses.

*Key Literature:*
* **Webb, B. (2019)** (*Neuron*): Survey on the cognitive architecture of insect navigation combining vector navigation (CX) and visual route memory (MB).
* **Müller et al. (2018) & Sun et al. (2020)** (*Science Robotics*): Dual-pathway insect navigation models proving that CX vector integration and MB snapshot memories synergize to navigate complex natural environments without GPS.
* **Nowotny et al. (2014–2021)**: Insect olfactory computation, sparse coding dynamics, and accelerated simulation in neuromorphic hardware.

---

### 2.2 Neuromorphic Computing & Spiking Neural Networks (SNNs)

#### 2.2.1 Neuron & Synapse Models
Neuromorphic systems trade continuous-valued floating-point matrix multiplications for sparse, asynchronous binary events (spikes):

1. **Leaky Integrate-and-Fire (LIF) Dynamics:**
   $$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + R_m \sum_{j} W_{ij} S_j(t - d_{ij}) + I_{\text{ext}}(t)$$
   $$\text{If } V_i(t) \ge V_{\text{th}}, \quad S_i(t) = 1 \quad \text{and} \quad V_i(t^+) \leftarrow V_{\text{reset}}$$

2. **Spike-Timing-Dependent Plasticity (STDP):**
   $$\Delta W_{ij} = \begin{cases} A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t = t_{\text{post}} - t_{\text{pre}} > 0 \\ -A_- \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \end{cases}$$

3. **Reward-Modulated STDP (R-STDP / Three-Factor Learning):**
   $$\frac{de_{ij}(t)}{dt} = -\frac{e_{ij}(t)}{\tau_e} + \text{STDP}(t), \qquad \frac{dW_{ij}(t)}{dt} = \eta \cdot M(t) \cdot e_{ij}(t)$$
   where $e_{ij}(t)$ is the synaptic eligibility trace and $M(t)$ is the neuromodulatory dopamine/octopamine concentration.

---

### 2.3 State-of-the-Art in Neuromorphic Silicon

| Platform | Architecture Type | Plasticity Support | Interconnect Topology | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Intel Loihi 1/2** | Asynchronous digital LIF | Microcode-programmable learning engine | 2D Mesh NoC (AER) | General-purpose neuromorphic research |
| **SpiNNaker 1/2** | Massively parallel ARM cores | Software-defined (on ARM CPUs) | Torus network with packet multicast | Large-scale brain simulation |
| **BrainScaleS-2** | Physical mixed-signal analog | Embedded SIMD plasticity processor | Hierarchical spatial crossbars | Accelerated (1000x) biophysics |
| **DYNAP-SE2** | Mixed-signal sub-threshold analog | Hardwired analog STDP | Hierarchical 2D routing | Ultra-low power edge sensing |
| **Tianjic** | Hybrid Deep Learning + SNN Digital ASIC | Off-chip / static weights | Crossbar Array + Global Router | Hybrid AGI robotics |
| **Proposed Ant-Chip** | **Dedicated bio-structural Digital ASIC** | **Hardware-embedded 3-Factor R-STDP & Ring-Attractor Engine** | **Hierarchical Columnar AER + Dedicated Ring-Shift Interconnect** | **Ultra-low latency sub-mW insect navigation** |

---

### 2.4 Identified Research Gaps
1. **Generic vs. Bio-Specialized Neuromorphic Topology:** General-purpose SNN chips (Loihi, SpiNNaker) allocate heavy silicon area to general-purpose mesh routing. The insect CX and MB possess **highly structured, sparse, columnar connectivity** (e.g., 8–16 periodic columns with shift-registers for ring attractors) that can be implemented in silicon with orders-of-magnitude less overhead.
2. **Online Plasticity with Zero Drift:** Existing digital insect models drift over prolonged foraging trajectories. Integrating three-factor eligibility traces with homeostatic synaptic scaling on-chip to continually calibrate heading and speed represents an unsolved problem.
3. **End-to-End Open-Source Silicon Verification:** Very few insect-inspired models are carried through from bio-simulation all the way to synthesizable register-transfer level (RTL) and open-source GDSII ASIC generation.

---

## 3. Research Objectives & Working Hypotheses

### 3.1 Primary Research Objectives
* **Objective 1 (Neuro-Computational Model):** Build a unified, spiking model of the desert ant (*Cataglyphis*) navigation system incorporating the Central Complex (CX) for vector path integration and the Mushroom Body (MB) for visual landmark recognition.
* **Objective 2 (Plasticity & Drift Correction):** Implement local, reward-modulated three-factor learning rules capable of one-shot landmark memory acquisition and online compass calibration without external backpropagation.
* **Objective 3 (Embodied Benchmark Evaluation):** Validate closed-loop navigation performance under environmental perturbations (wind drift, visual occlusion, motor noise, dynamic obstacle fields) against state-of-the-art DRL and classical SLAM baselines.
* **Objective 4 (Digital Silicon Microarchitecture):** Design an asynchronous/event-driven digital neuromorphic accelerator in synthesizable SystemVerilog/Verilog, characterized by cycle-accurate simulation, profiling latency, energy per synaptic operation ($\text{pJ/SOP}$), and gate count.
* **Objective 5 (Open Silicon & Academic Dissemination):** Harden the architecture using the SkyWater 130nm / OpenLane flow and publish findings in top-tier peer-reviewed journals and conferences.

### 3.2 Formal Hypotheses
* **$\mathbf{H_1}$ (Computational Efficiency):** An ant-inspired SNN architecture will achieve identical or superior path integration accuracy to a deep recurrent LSTM baseline while reducing total synaptic compute operations by $>90\%$ due to event sparsity ($<5\%$ active neurons per timestep).
* **$\mathbf{H_2}$ (Drift Resistance):** Synergistic coupling of MB visual snapshot matching with CX vector steering will eliminate cumulative dead-reckoning drift across journeys exceeding $10^4$ steps without requiring global map reconstruction.
* **$\mathbf{H_3}$ (Silicon Area & Power Scaling):** Exploiting insect-specific ring-attractor columnar connectivity in a dedicated digital ASIC will yield an energy efficiency $< 0.5\text{ pJ per synaptic event}$ at $1.2\text{V}$, outperforming general-purpose edge GPUs by $> 1000\times$.

---

## 4. System Architecture & Technical Methodology

```
+---------------------------------------------------------------------------------------------------------+
|                                    FIVE-LAYER R&D SYSTEM PIPELINE                                       |
+---------------------------------------------------------------------------------------------------------+
|  Layer 1: Bio-Accurate Mathematical Equations (LIF, Ring Dynamics, Vector Arithmetic)                  |
|                                                    |                                                    |
|                                                    v                                                    |
|  Layer 2: Python SNN Simulation & Benchmark Suite (SpikingJelly / Brian2 / snnTorch)                    |
|                                                    |                                                    |
|                                                    v                                                    |
|  Layer 3: Closed-Loop Embodied Simulation Arena (Gym-Insect 2D/3D Navigation Arena)                     |
|                                                    |                                                    |
|                                                    v                                                    |
|  Layer 4: Hardware Description (SystemVerilog RTL: AER Router + LIF Core + SRAM Crossbar)               |
|                                                    |                                                    |
|                                                    v                                                    |
|  Layer 5: Digital Verification & Synthesis (Verilator / Cocotb / OpenLane SkyWater 130nm)               |
+---------------------------------------------------------------------------------------------------------+
```

### 4.1 Mathematical Formulation of the Ant Navigation Circuit

#### 4.1.1 Protocerebral Bridge (PB) Compass Ring Attractor
The heading compass network is composed of $N = 8$ (or 16) discrete azimuthal columns. Let $\theta_i = \frac{2\pi i}{N}$ be the preferred heading of column $i$.
The membrane dynamics of compass neuron $u_i(t)$ follow a discrete-time leaky integrator with recurrent lateral inhibition and cosine-tuned excitation:

$$u_i(t + \Delta t) = (1 - \alpha_c) u_i(t) + \alpha_c \left[ \sum_{j=1}^N W_{ij}^{\text{ring}} S_j(t) + I_i^{\text{DRA}}(\theta_{\text{celestial}}) + I_{\text{shift}, i}(\Delta \theta) \right]$$

where:
$$W_{ij}^{\text{ring}} = w_{\text{exc}} \cos(\theta_i - \theta_j) - w_{\text{inh}}$$
and $I_{\text{shift}, i}(\Delta \theta)$ represents angular velocity inputs from Noduli-innervating neurons, rotating the activity bump left or right when the agent turns.

#### 4.1.2 Fan-Shaped Body (FB) Path Integrator Neurons
CPU4 (or PF-Col) accumulator neurons maintain the home displacement vector. Each neuron $v_i(t)$ receives excitation from its corresponding compass column $i$ gated by forward speed $v_{\text{speed}}(t)$:

$$\tau_a \frac{dv_i(t)}{dt} = -v_i(t) + \beta \cdot S_{\text{compass}, i}(t) \cdot v_{\text{speed}}(t) - \gamma \cdot S_{\text{reset}}(t)$$

When returning home, the accumulated vector is read out through bridge neurons (CPU1) to compute a turning command $\Delta \phi$:
$$\text{Steering Bias } S_{\text{steer}} = \sum_{i=1}^{N/2} v_i^{\text{Left}}(t) - \sum_{i=1}^{N/2} v_i^{\text{Right}}(t)$$

#### 4.1.3 Mushroom Body (MB) Visual Landmark Matching
Visual scenes are encoded into binary edge vectors $\vec{x}_{\text{retina}} \in \{0, 1\}^D$.
1. **Random Projection to Kenyon Cells:**
   $$\vec{z}_{\text{KC}} = \Theta\left( \mathbf{W}_{\text{rand}} \vec{x}_{\text{retina}} - \vec{\theta}_{\text{KC}} \right), \quad \text{where } \|\vec{z}_{\text{KC}}\|_0 \ll \dim(\vec{z}_{\text{KC}})$$
2. **Familiarity Scoring at MBON:**
   $$\text{Valence / Familiarity } F = \mathbf{W}_{\text{MBON}}^T \vec{z}_{\text{KC}}$$
3. **Synaptic Weight Update (Anti-Hebbian / LTD upon rewarded goal reach):**
   $$\Delta \mathbf{W}_{\text{MBON}} = -\eta \cdot R(t) \cdot \vec{z}_{\text{KC}}$$

---

### 4.2 Digital Neuromorphic Chip Microarchitecture

```
+--------------------------------------------------------------------------------------------------+
|                            DIGITAL NEUROMORPHIC CORE ARCHITECTURE                                |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +--------------------------+       +-------------------------------+                           |
|   |   AER Ingress Router     | ----> | Spike Decoder & Event FIFO    |                           |
|   |  (Address-Event Packet)  |       +-------------------------------+                           |
|   +--------------------------+                      |                                            |
|                                                     v                                            |
|   +---------------------------------------------------------------------+                        |
|   |                      Configurable Synaptic Crossbar                 |                        |
|   |  - Static Weights (ROM / Register File) for Ring Attractors         |                        |
|   |  - Dynamic Plastic Weights (Dual-Port SRAM) for Mushroom Body       |                        |
|   |  - Synapse Accumulator Matrix: 16-bit Fixed-Point Q4.12             |                        |
|   +---------------------------------------------------------------------+                        |
|                                                     |                                            |
|                                                     v                                            |
|   +---------------------------------------------------------------------+                        |
|   |                      Vectorized LIF Processing Core                 |                        |
|   |  - Leak Multiplier / Subtractor: V_m[i] <= V_m[i] * (1 - lambda)    |                        |
|   |  - Threshold Comparator: V_m[i] >= V_th                             |                        |
|   |  - Refractory Counter Logic                                         |                        |
|   |  - Spike Generation Unit (Binary Mask & Event Encoder)              |                        |
|   +---------------------------------------------------------------------+                        |
|                     |                                       |                                    |
|                     v                                       v                                    |
|   +------------------------------------+  +------------------------------------+                 |
|   | 3-Factor R-STDP Plasticity Engine  |  | AER Egress Packetizer              |                 |
|   | (Eligibility trace update & decay) |  | (Spike output to motor / mesh NoC) |                 |
|   +------------------------------------+  +------------------------------------+                 |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

#### 4.2.1 Microarchitecture Specifications
1. **Arithmetic Precision:** Q4.12 Fixed-Point (16-bit signed: 1 sign, 3 integer bits, 12 fractional bits). Eliminates all floating-point units (FPUs) to minimize silicon area.
2. **Neuron Processing Units (NPUs):** Time-multiplexed SIMD execution pipeline processing 16 LIF neurons per clock cycle.
3. **Memory Organization:**
   * **CX Hardwired Core:** Hardwired routing matrix in registers/LUTs for ring-attractor shifts (zero SRAM lookup overhead).
   * **MB Plastic Synapse Memory:** Local 16KB dual-port high-density SRAM for KC-to-MBON plastic connections.
4. **Asynchronous/Synchronous Interconnect:** Address-Event Representation (AER) packet encoding:
   $$\text{AER Packet} = [\text{Core\_ID (4 bits)} \,|\, \text{Neuron\_ID (10 bits)} \,|\, \text{Timestamp (10 bits)} \,|\, \text{Spike\_Val (1 bit)}]$$

---

## 5. Experimental Design, Benchmarks & Metrics

### 5.1 Experimental Protocols

| Experiment ID | Description | Benchmark Objective | Baselines |
| :--- | :--- | :--- | :--- |
| **EXP-1** | Pure Path Integration in High-Noise Arena | Test zero-drift heading tracking and homing accuracy over 5,000 steps | Standard LIF SNN, LSTM Recurrent Controller, Classical Odometry |
| **EXP-2** | Wind Drift & Forced Displacement | Evaluate homing vector recovery after forced passive displacement | Uncoupled CX vs. Coupled CX+MB |
| **EXP-3** | Cluttered Panoramic Visual Navigation | Route following and nest localization in 3D visually realistic arena | CNN-based Visual SLAM (ORB-SLAM3), Supervised ResNet |
| **EXP-4** | RTL Hardware Cycle-Accurate Emulation | Profile spike latency, throughput, and memory bandwidth in Verilator | Intel Loihi-2 (simulated), SpiNNaker-2 |
| **EXP-5** | ASIC Physical Synthesis (SkyWater 130nm) | Measure total silicon area ($\text{mm}^2$), power ($\mu\text{W}$), and Energy/SOP | Edge Tensor Processors, Microcontrollers (ARM Cortex-M4) |

### 5.2 Quantitative Evaluation Metrics
* **Navigational Accuracy:**
  * **Homing Error ($\epsilon_{\text{home}}$):** Euclidean distance between agent's final homing point and actual nest origin:
    $$\epsilon_{\text{home}} = \|\vec{x}_{\text{final}} - \vec{x}_{\text{nest}}\|_2$$
  * **Path Tortuosity Index ($\tau$):** Ratio of actual path length to straight-line distance:
    $$\tau = \frac{L_{\text{actual}}}{D_{\text{euclidean}}} \ge 1.0$$
* **Neuromorphic & Silicon Metrics:**
  * **Synaptic Operations (SynOps):** Count of weight-accumulation events per second.
  * **Energy per Synaptic Operation ($E_{\text{SOP}}$):** $\text{pJ / SOP} = \frac{\text{Total Dynamic Power}}{\text{SynOps / sec}}$.
  * **Area Efficiency:** Number of configurable neurons and synapses per $\text{mm}^2$ of silicon.

---

## 6. PhD Project Milestones & Timeline

```
+-------------------------------------------------------------------------------------------------------+
|                                    3-YEAR PHD TIMELINE & MILESTONES                                   |
+-------------------------------------------------------------------------------------------------------+
| [Month 01 - 06] PHASE 1: Mathematical Formulation & Python SNN Baseline                               |
|                 - Implement biologically accurate CX (PB, EB, FB, NO) in Python (SpikingJelly/Brian2) |
|                 - Validate ring-attractor stability and velocity accumulation                         |
|                 - Deliverable: Conference Paper #1 (e.g., NeurIPS / ICLR NeuroAI Track)               |
+-------------------------------------------------------------------------------------------------------+
| [Month 07 - 14] PHASE 2: Embodied Simulation Arena & Mushroom Body Memory Coupling                    |
|                 - Build 2D/3D foraging arena with celestial polarization and visual snapshots        |
|                 - Integrate 3-factor R-STDP for one-shot landmark association                         |
|                 - Deliverable: Journal Paper #1 (e.g., Science Robotics / IEEE T-NNLS)                |
+-------------------------------------------------------------------------------------------------------+
| [Month 15 - 24] PHASE 3: Digital Neuromorphic Core Design (SystemVerilog RTL)                         |
|                 - Implement Q4.12 fixed-point LIF NPU cores, AER packet router, and SRAM controller  |
|                 - Build cycle-accurate Verilator C++ testbench and verify bit-accuracy against SNN    |
|                 - Deliverable: Conference Paper #2 (e.g., IEEE ISCAS / ICONS)                         |
+-------------------------------------------------------------------------------------------------------+
| [Month 25 - 32] PHASE 4: ASIC Hardening & Open-Source Silicon Synthesis                               |
|                 - Run OpenLane / SkyWater 130nm physical design flow (GDSII layout generation)        |
|                 - Extract parasitic RC, perform static timing analysis (STA) & power profiling       |
|                 - Deliverable: Journal Paper #2 (e.g., IEEE Transactions on CAS-I / TCAD)             |
+-------------------------------------------------------------------------------------------------------+
| [Month 33 - 36] PHASE 5: Doctoral Dissertation Defense & Open-Source Benchmark Release                |
|                 - Release open-source `AntNeuromorphic` simulator and synthesizable Verilog repository|
|                 - Complete final PhD Thesis dissertation                                              |
+-------------------------------------------------------------------------------------------------------+
```

---

## 7. Target Publication Venues & Impact Strategy

### 7.1 Tier-1 Target Journals
1. **Science Robotics / Nature Communications:** Highlighting bio-inspired autonomous navigation, zero-drift spatial intelligence, and neuromorphic efficiency.
2. **IEEE Transactions on Neural Networks and Learning Systems (TNNLS):** Emphasizing mathematical formulations of SNN ring attractors and R-STDP online plasticity.
3. **IEEE Transactions on Circuits and Systems I (TCAS-I):** Focused on digital neuromorphic circuit design, AER routing microarchitecture, and silicon power profiling.

### 7.2 Top International Conferences
1. **NeurIPS / ICLR / ICRA:** NeuroAI / Bio-Inspired Robotics tracks.
2. **IEEE International Symposium on Circuits and Systems (ISCAS):** Neuromorphic circuit design & VLSI implementations.
3. **International Conference on Neuromorphic Systems (ICONS):** Core neuromorphic algorithms and hardware benchmarks.

---

## 8. Foundational References & Academic Bibliography

1. **Stone, T., Webb, B., Adden, A., Weddig, N. B., Honkanen, A., Templin, R., ... & Heinze, S. (2017).** An anatomically constrained model for path integration in the bee brain. *Current Biology*, 27(20), 3069-3085.
2. **Webb, B. (2019).** The internal maps of insects. *Neuron*, 102(5), 911-913.
3. **Sun, X., Yue, S., & Mangan, M. (2020).** A decentralized neural model explaining optimal integration of visual navigation and path integration in insects. *Science Robotics*, 5(43), eaaz8611.
4. **Davies, M., Srinivasa, N., Lin, T. H., Chinya, G., Cao, Y., Choday, S. H., ... & Wang, H. (2018).** Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82-99.
5. **Heinze, S. (2017).** Unraveling the neural basis of insect navigation. *Current Opinion in Insect Science*, 24, 58-67.
6. **Müller, M., & Wehner, R. (1988).** Path integration in desert ants, *Cataglyphis fortis*. *Proceedings of the National Academy of Sciences*, 85(14), 5287-5290.
7. **Nowotny, T., Huerta, R., Abarbanel, H. D., & Rabinovich, M. I. (2005).** Self-organization in the olfactory system: one-shot associative learning in a neurocomputational model. *Biological Cybernetics*, 93(6), 436-446.
8. **Frenkel, C., Lefebvre, M., Bol, D., & Legat, J. D. (2019).** MorphIC: A 2.4-pJ/synaptic-ops 1024-neuron 130-nm SNN processor with digital event-driven online learning. *IEEE Transactions on Biomedical Circuits and Systems*, 13(5), 999-1010.
9. **Indiveri, G., Linares-Barranco, B., Hamilton, T. J., Schaik, A. V., Etienne-Cummings, R., Delbruck, T., ... & Boahen, K. (2011).** Neuromorphic silicon neuron circuits. *Frontiers in Neuroscience*, 5, 73.
10. **Wystrach, A., Schwarz, S., Schultheiss, P., Beugnon, G., & Cheng, K. (2011).** Views, landmarks, and routes: how do desert ants navigate?. *Journal of Comparative Physiology A*, 197(9), 927-943.
