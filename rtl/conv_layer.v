module conv_layer #(
    parameter IMG_W      = 28,
    parameter IMG_H      = 28,
    parameter N_FILTERS  = 8,
    parameter N_CHANNELS = 1
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       pixel_valid,
    input  wire [7:0]                 pixel_in,          // single channel input
    input  wire [N_CHANNELS*8-1:0]    data_in,           // multi-channel input
    output reg                        out_valid,
    output reg  [N_FILTERS*8-1:0]     conv_out
);

    // Line buffers
    reg signed [7:0] line_buf0 [0:N_CHANNELS*IMG_W-1];
    reg signed [7:0] line_buf1 [0:N_CHANNELS*IMG_W-1];
    reg signed [7:0] shift_r   [0:N_CHANNELS*3-1];

    // 3x3 window values
    wire signed [7:0] win_flat [0:N_CHANNELS*9-1];
    genvar gch;
    generate
        for (gch = 0; gch < N_CHANNELS; gch = gch + 1) begin : gen_win
            assign win_flat[gch*9+0] = line_buf0[gch*IMG_W+2];
            assign win_flat[gch*9+1] = line_buf0[gch*IMG_W+1];
            assign win_flat[gch*9+2] = line_buf0[gch*IMG_W+0];
            assign win_flat[gch*9+3] = line_buf1[gch*IMG_W+2];
            assign win_flat[gch*9+4] = line_buf1[gch*IMG_W+1];
            assign win_flat[gch*9+5] = line_buf1[gch*IMG_W+0];
            assign win_flat[gch*9+6] = shift_r[gch*3+2];
            assign win_flat[gch*9+7] = shift_r[gch*3+1];
            assign win_flat[gch*9+8] = shift_r[gch*3+0];
        end
    endgenerate

    // Kernel memory
    localparam ROM_DEPTH = N_FILTERS * N_CHANNELS * 9;
    reg signed [7:0] kernel_rom [0:ROM_DEPTH-1];

    initial begin
        if (N_CHANNELS == 1)
            $readmemh("conv1_weights.hex", kernel_rom);
        else
            $readmemh("conv2_weights.hex", kernel_rom);
    end

    localparam NTAPS = N_CHANNELS * 9;

    // Multiply and accumulate
    wire signed [15:0] dp_prod [0:N_FILTERS-1][0:NTAPS-1];
    wire signed [23:0] dp_psh  [0:N_FILTERS-1][0:NTAPS-1];
    wire signed [23:0] dp_sum  [0:N_FILTERS-1][0:NTAPS-1];

    genvar gf, gt;
    generate
        for (gf = 0; gf < N_FILTERS; gf = gf + 1) begin : gen_f
            for (gt = 0; gt < NTAPS; gt = gt + 1) begin : gen_t

                // Multiply input with weight
                assign dp_prod[gf][gt] =
                    win_flat[gt] * kernel_rom[gf * NTAPS + gt];

                // Scale result
                assign dp_psh[gf][gt] =
                    {{8{dp_prod[gf][gt][15]}}, dp_prod[gf][gt]} >>> 3;

                // Accumulate
                if (gt == 0)
                    assign dp_sum[gf][0] = dp_psh[gf][0];
                else
                    assign dp_sum[gf][gt] = dp_sum[gf][gt-1] + dp_psh[gf][gt];
            end
        end
    endgenerate

    // Final sum for each filter
    wire signed [23:0] filter_acc [0:N_FILTERS-1];

    genvar ga;
    generate
        for (ga = 0; ga < N_FILTERS; ga = ga + 1) begin : gen_acc
            assign filter_acc[ga] = dp_sum[ga][NTAPS-1];
        end
    endgenerate

    // Position counters
    reg [$clog2(IMG_W)-1:0] col_cnt;
    reg [$clog2(IMG_H)-1:0] row_cnt;
    reg [1:0]               rows_loaded;

    // Window ready signal
    wire window_valid = pixel_valid
                        && (rows_loaded == 2'b10)
                        && (col_cnt >= 2);

    // Main logic
    integer i, m;
    always @(posedge clk) begin
        if (!rst_n) begin
            col_cnt     <= 0;
            row_cnt     <= 0;
            rows_loaded <= 0;
            out_valid   <= 1'b0;

        end else if (pixel_valid) begin

            // Shift data for each channel
            for (m = 0; m < N_CHANNELS; m = m + 1) begin

                shift_r[m*3+2] <= shift_r[m*3+1];
                shift_r[m*3+1] <= shift_r[m*3+0];

                shift_r[m*3+0] <= (N_CHANNELS == 1)
                                  ? $signed(pixel_in)
                                  : $signed(data_in[m*8 +: 8]);

                // Shift line buffers
                for (i = IMG_W-1; i > 0; i = i-1) begin
                    line_buf0[m*IMG_W+i] <= line_buf0[m*IMG_W+i-1];
                    line_buf1[m*IMG_W+i] <= line_buf1[m*IMG_W+i-1];
                end

                // Move old data between buffers
                line_buf1[m*IMG_W+0] <= shift_r[m*3+2];
                line_buf0[m*IMG_W+0] <= line_buf1[m*IMG_W + IMG_W-1];
            end

            // Update image position
            if (col_cnt == IMG_W-1) begin
                col_cnt <= 0;

                row_cnt <= (row_cnt == IMG_H-1)
                           ? 0
                           : row_cnt + 1;

                if (rows_loaded < 2'b10)
                    rows_loaded <= rows_loaded + 1;

            end else begin
                col_cnt <= col_cnt + 1;
            end

            // Output result
            out_valid <= window_valid;

            if (window_valid) begin
                for (i = 0; i < N_FILTERS; i = i+1) begin

                    // ReLU
                    if (filter_acc[i][23])
                        conv_out[i*8 +: 8] <= 8'h00;

                    // Saturation
                    else if (|filter_acc[i][23:11])
                        conv_out[i*8 +: 8] <= 8'h7F;

                    // Normal output
                    else
                        conv_out[i*8 +: 8] <= filter_acc[i][10:3];
                end
            end

        end else begin
            out_valid <= 1'b0;
        end
    end

endmodule
