module mnist_cnn_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        pixel_valid,
    input  wire [7:0]  pixel_raw,
    output wire [3:0]  digit_out,
    output wire        result_valid
);

    // Convert pixel to Q4.3 format
    wire [7:0] pixel_norm = {1'b0, pixel_raw[7:5], 3'b000};

    // Conv1: 28x28x1 -> 26x26x8
    wire        conv1_valid;
    wire [63:0] conv1_out;

    conv_layer #(
        .IMG_W      (28),
        .IMG_H      (28),
        .N_FILTERS  (8),
        .N_CHANNELS (1)
    ) u_conv1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .pixel_valid(pixel_valid),
        .pixel_in   (pixel_norm),
        .data_in    (8'h00),   // unused
        .out_valid  (conv1_valid),
        .conv_out   (conv1_out)
    );

    // Pool1: 26x26x8 -> 13x13x8
    wire        pool1_valid;
    wire [63:0] pool1_out;

    max_pool_layer #(
        .N_MAPS(8),
        .IMG_W (26),
        .IMG_H (26)
    ) u_pool1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (conv1_valid),
        .data_in  (conv1_out),
        .out_valid(pool1_valid),
        .data_out (pool1_out)
    );

    // Conv2: 13x13x8 -> 11x11x16
    wire         conv2_valid;
    wire [127:0] conv2_out;

    conv_layer #(
        .IMG_W      (13),
        .IMG_H      (13),
        .N_FILTERS  (16),
        .N_CHANNELS (8)
    ) u_conv2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .pixel_valid(pool1_valid),
        .pixel_in   (8'h00),   // unused
        .data_in    (pool1_out),
        .out_valid  (conv2_valid),
        .conv_out   (conv2_out)
    );

    // Pool2: 11x11x16 -> 5x5x16
    wire         pool2_valid;
    wire [127:0] pool2_out;

    max_pool_layer #(
        .N_MAPS(16),
        .IMG_W (11),
        .IMG_H (11)
    ) u_pool2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (conv2_valid),
        .data_in  (conv2_out),
        .out_valid(pool2_valid),
        .data_out (pool2_out)
    );

    // Fully Connected Layer
    wire [3:0] fc_digit;
    wire       fc_valid;

    fc_layer #(
        .IN_SPATIAL(25),
        .N_CH      (16),
        .IN_SIZE   (400),
        .OUT_SIZE  (10)
    ) u_fc (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (pool2_valid),
        .data_in     (pool2_out),
        .digit_out   (fc_digit),
        .result_valid(fc_valid)
    );

    // Final outputs
    assign digit_out    = fc_digit;
    assign result_valid = fc_valid;

endmodule
