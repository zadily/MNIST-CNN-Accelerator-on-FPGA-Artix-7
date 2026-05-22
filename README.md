# MNIST-CNN-Accelerator-on-FPGA-Artix-7
A fully custom CNN inference engine built in Verilog-2001, targeting the Xilinx Artix-7 (Arty A7-35T). Recognises handwritten digits (0–9) from the MNIST dataset using a streaming pixel pipeline - no floating point, no HLS, pure RTL. Weights are trained in PyTorch, quantised to Q4.3 fixed-point, and loaded directly into on-chip BRAM. 

# MNIST CNN Accelerator on FPGA

> Full CNN inference pipeline in pure RTL — no HLS, no floating point. Recognises handwritten digits 0–9 from a live pixel stream.

---

## Table of Contents
- [About the Project](#about-the-project)
  - [Tech Stack](#tech-stack)
  - [File Structure](#file-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [System Architecture](#system-architecture)
- [Fixed-Point Quantisation](#fixed-point-quantisation)
- [Training and Weight Export](#training-and-weight-export)
- [Simulation](#simulation)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## About the Project

This project implements a **Convolutional Neural Network accelerator** entirely in synthesisable Verilog-2001, targeting the Xilinx Artix-7 FPGA. A 28×28 grayscale image is streamed in one pixel per clock, passing through two convolution layers, two max-pool layers, and a fully connected classifier — producing a digit prediction in ~40 µs at 100 MHz.

Weights are trained in PyTorch, quantised to 8-bit Q4.3 fixed-point, and loaded into on-chip ROM via `$readmemh`. The entire 4,072-weight model lives in FPGA fabric with no external memory.

**Project Specifications:**
- **Target Platform:** Arty A7-35T (Xilinx Artix-7)
- **Primary Domains:** FPGA Design, Neural Network Inference, Fixed-Point Arithmetic
- **Inference Latency:** ~40 µs @ 100 MHz
- **Model Accuracy:** 97.6% on MNIST test set (Q4.3 quantised)

### Tech Stack
- **Verilog-2001** — RTL design, synthesisable on all Vivado versions
- **Xilinx Vivado** — Synthesis, implementation, simulation
- **PyTorch** — Model training and weight quantisation
- **Python** — Weight export and software golden model

### File Structure
```
mnist-cnn-fpga/
│
├── rtl/
│   ├── mac_unit.v              # Multiply-accumulate unit → DSP48
│   ├── max_pool_2x2.v          # Combinational 2×2 max, single channel
│   ├── max_pool_layer.v        # Streaming pool, N_MAPS channels
│   ├── conv_layer.v            # Streaming conv + ReLU, parameterised
│   ├── fc_layer.v              # FC layer + argmax FSM
│   └── mnist_cnn_top.v         # Top-level pipeline
│
├── tb/
│   └── mnist_cnn_tb.v          # Full pipeline testbench
│
├── weights/
│   ├── conv1_weights.hex       # 8×1×3×3  = 72 weights
│   ├── conv2_weights.hex       # 16×8×3×3 = 1152 weights
│   └── fc_weights.hex          # 10×400   = 4000 weights
│
├── python/
│   ├── train_and_export.py     # Train PyTorch model, export .hex
│   └── golden_model.py         # Q4.3 software reference model
│
├── test_images/                # Sample 28×28 hex input images
│   ├── digit1_test.hex
│   ├── digit3_test.hex
│   ├── digit5_test.hex
│   ├── digit7_test.hex
│   ├── digit8_test.hex
│   └── digit9_test.hex
│
└── constraints/
    └── arty_a7.xdc
```

---

## Getting Started

### Prerequisites

- **Xilinx Vivado** 2019.1 or later (WebPACK edition sufficient)
- **Python 3.8+** with PyTorch and torchvision
- **Arty A7-35T** development board *(for hardware deployment)*

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/mnist-cnn-fpga.git
cd mnist-cnn-fpga
```

#### 2. Train the model and export weights
```bash
pip install torch torchvision numpy
python python/train_and_export.py
# Outputs: weights/conv1_weights.hex, conv2_weights.hex, fc_weights.hex
```

#### 3. Verify with the software golden model
```bash
python python/golden_model.py
# Prints Q4.3 logits and predicted digit — use to cross-check simulation
```

#### 4. Create Vivado project
1. Launch Vivado → **Create Project** → RTL Project
2. Target device: `xc7a35ticsg324-1L`
3. Add all files from `rtl/` as design sources
4. Add `tb/mnist_cnn_tb.v` as simulation source
5. Copy `weights/*.hex` and a test image from `test_images/` into the project directory

#### 5. Run simulation
1. Set `mnist_cnn_tb` as simulation top
2. **Run Simulation → Run Behavioural Simulation**
3. Check the Tcl console for `[MON]` pipeline output

#### 6. Deploy to hardware *(optional)*
1. Set `mnist_cnn_top` as synthesis top
2. Add `constraints/arty_a7.xdc`
3. **Run Implementation → Generate Bitstream**
4. Open Hardware Manager → Auto Connect → Program Device

---

## System Architecture


| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| Conv1 | 3×3 conv + ReLU | 28×28×1 | 26×26×8 |
| Pool1 | 2×2 max pool | 26×26×8 | 13×13×8 |
| Conv2 | 3×3 conv + ReLU | 13×13×8 | 11×11×16 |
| Pool2 | 2×2 max pool | 11×11×16 | 5×5×16 |
| FC | Dense + argmax | 400 | 10 |


### Key Design Points

**Line buffer sliding window** — Two shift-register rows per channel maintain the 3×3 neighbourhood as pixels stream in. No frame buffer required.

**Stall / valid handshake** — Each conv layer asserts `stall_out` for 9 cycles during MAC computation, pausing the upstream pixel stream so the line buffer does not shift mid-accumulation.

**ReLU at zero cost** — Implemented as a single conditional on the accumulator sign bit. No extra LUTs.

**FC FSM** — Three states: `FILL` (collect 400 inputs), `COMPUTE` (4,000 MAC cycles across 10 neurons), `ARGMAX` (scan 10 logits, output winner).

> **DSP note:** The Artix-7 35T has 90 DSP48 slices; this design needs ~216. Reduce to 4/8 filters for 35T deployment, or target the Artix-7 100T (240 DSPs) / Zynq-7020 (220 DSPs) for the full design.

---

## Fixed-Point Quantisation

All weights and activations use **Q4.3** — 1 sign bit, 4 integer bits, 3 fraction bits packed into a signed 8-bit register.

```
Bit:  7    6    5    4  |  3    2    1    0
      S    2³   2²   2¹    2⁰   2⁻¹  2⁻²  2⁻³
                        ↑ binary point

Range: −8.000 to +7.875     Step: 0.125
```

Multiplying two Q4.3 values gives a 16-bit Q8.6 product — re-quantised back to Q4.3 with an arithmetic right shift by 3:

```verilog
// WRONG — concatenation {} is always unsigned; >>> fills zeros, not sign bit
wire signed [19:0] x = {{4{p[15]}}, p} >>> 3;

// CORRECT — $signed() forces arithmetic right shift
wire signed [19:0] x = $signed({{4{p[15]}}, p}) >>> 3;
```

> This is one of the most common silent bugs in fixed-point Verilog. All shift operations in this design use the `$signed()` cast.

---

## Training and Weight Export

`train_and_export.py` trains on MNIST for 5 epochs (~98% float32 accuracy), quantises weights to Q4.3, and writes one weight per line as two-digit hex:

```python
def quantise_q43(tensor):
    return torch.clamp(torch.round(tensor * 8), -128, 127).to(torch.int8)
```

| File | Shape | Entries |
|------|-------|---------|
| `conv1_weights.hex` | [8, 1, 3, 3] | 72 |
| `conv2_weights.hex` | [16, 8, 3, 3] | 1152 |
| `fc_weights.hex` | [10, 400] | 4000 |

`golden_model.py` re-runs inference using the same Q4.3 arithmetic entirely in Python — compare its predicted digit with the Vivado `[MON]` output to confirm hardware correctness.

---

## Simulation

**Expected Tcl console output:**
```
[MON] conv1 first output at t=725000    val0=xx
[MON] pool1 first output at t=3455000
[MON] conv2 output at t=...
[MON] pool2 output at t=...
[MON] result_valid! digit=1  t=103515000
```

**Key waveform signals:**

| Signal | What to look for |
|--------|-----------------|
| `dut.u_conv1.out_valid` | Fires after 2 full rows loaded (~58 cycles) |
| `dut.conv1_stall` | 9-cycle pulses during MAC computation |
| `dut.u_fc.state` | 0=FILL → 1=COMPUTE → 2=ARGMAX |
| `result_valid` | Single-cycle pulse when digit is ready |
| `digit_out[3:0]` | Should match `golden_model.py` output |

**Input image format** — 784 lines, one byte per line, row-major, top-left first:
```
00        ← black pixel  →  Q4.3 = 0
ff        ← white pixel  →  Q4.3 = 7
dc        ← 220 decimal  →  Q4.3 = 6
```

---

## Future Enhancements

- **AXI-Stream interface** — Replace valid/stall handshake with AXI4-Stream for Zynq PS/PL integration
- **BRAM weight storage** — Move weight ROMs from distributed LUT RAM into block RAM to free LUT resources
- **More digits / full demo** — Connect OV7670 camera module and 7-segment display for a live recognition demo
- **Batch normalisation** — Add per-channel scale/offset for improved post-quantisation accuracy

---

## Contributors

- [Zaid Faruqui](https://github.com/zadily)
