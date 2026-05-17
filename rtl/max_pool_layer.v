module max_pool_layer #(
    parameter N_MAPS = 8,
    parameter IMG_W  = 26,
    parameter IMG_H  = 26
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [N_MAPS*8-1:0]  data_in,    // packed feature maps
    output reg                  out_valid,
    output reg  [N_MAPS*8-1:0]  data_out
);

    // Position counters
    reg [$clog2(IMG_W)-1:0] col;
    reg [$clog2(IMG_H)-1:0] row;

    // Line buffer for previous row
    reg [7:0] line_buf [0:N_MAPS*IMG_W-1];

    // Store nearby pixels for 2x2 window
    reg [7:0] prev_col [0:N_MAPS-1];
    reg [7:0] top_left [0:N_MAPS-1];

    wire [7:0] pool_out  [0:N_MAPS-1];
    wire [7:0] top_right [0:N_MAPS-1];

    // Pooling valid on odd row and odd column
    wire window_rdy = in_valid & col[0] & row[0];

    genvar m;
    generate
        for (m = 0; m < N_MAPS; m = m + 1) begin : gen_pool

            assign top_right[m] = line_buf[m*IMG_W + col];

            // 2x2 max pooling block
            max_pool_2x2 u_mp (
                .p00 (top_left[m]),
                .p01 (top_right[m]),
                .p10 (prev_col[m]),
                .p11 (data_in[m*8 +: 8]),
                .pout(pool_out[m])
            );
        end
    endgenerate

    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            col       <= 0;
            row       <= 0;
            out_valid <= 1'b0;

        end else if (in_valid) begin

            // Update buffers
            for (i = 0; i < N_MAPS; i = i + 1) begin

                prev_col[i] <= data_in[i*8 +: 8];

                top_left[i] <= line_buf[i*IMG_W + col];

                line_buf[i*IMG_W + col] <= data_in[i*8 +: 8];
            end

            // Update position
            if (col == IMG_W - 1) begin
                col <= 0;

                row <= (row == IMG_H - 1)
                       ? 0
                       : row + 1;

            end else begin
                col <= col + 1;
            end

            // Output pooled value
            out_valid <= window_rdy;

            if (window_rdy) begin
                for (i = 0; i < N_MAPS; i = i + 1)
                    data_out[i*8 +: 8] <= pool_out[i];
            end

        end else begin
            out_valid <= 1'b0;
        end
    end

endmodule
