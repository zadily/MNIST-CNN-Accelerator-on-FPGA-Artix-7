# MNIST-CNN-Accelerator-on-FPGA-Artix-7
A fully custom CNN inference engine built in Verilog-2001, targeting the Xilinx Artix-7 (Arty A7-35T). Recognises handwritten digits (0–9) from the MNIST dataset using a streaming pixel pipeline - no floating point, no HLS, pure RTL. Weights are trained in PyTorch, quantised to Q4.3 fixed-point, and loaded directly into on-chip BRAM. 
