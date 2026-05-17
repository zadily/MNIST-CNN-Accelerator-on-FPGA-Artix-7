module fc_layer #(
    parameter IN_SPATIAL = 25,
    parameter N_CH       = 16,
    parameter IN_SIZE    = 400,
    parameter OUT_SIZE   = 10,
    parameter DATA_W     = 8,
    parameter ACC_W      = 26
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  in_valid,
    input  wire [N_CH*8-1:0]     data_in,
    output reg  [3:0]            digit_out,
    output reg                   result_valid
);

    // Input buffer
    reg signed [DATA_W-1:0] input_buf [0:IN_SIZE-1];

    reg [$clog2(IN_SPATIAL):0] buf_pos;

    // Weight memory
    reg signed [DATA_W-1:0] fc_w [0:OUT_SIZE*IN_SIZE-1];
    initial $readmemh("fc_weights.hex", fc_w);

    // FSM states
    localparam S_FILL    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_ARGMAX  = 2'd2;

    reg [1:0] state;

    reg [$clog2(IN_SIZE)+1:0]  i_cnt;
    reg [$clog2(OUT_SIZE):0]   n_cnt;

    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] logits [0:OUT_SIZE-1];

    reg signed [ACC_W-1:0] max_val;
    reg [3:0]              max_idx;

    reg [$clog2(OUT_SIZE):0] arg_cnt;

    reg signed [17:0] a_reg, b_reg;

    // MAC intermediate values
    reg signed [15:0]      mac_prod;
    reg signed [ACC_W-1:0] mac_shift;

    integer ch;

    always @(posedge clk) begin
        if (!rst_n) begin

            buf_pos      <= 0;
            state        <= S_FILL;

            result_valid <= 1'b0;

            i_cnt        <= 0;
            n_cnt        <= 0;

            acc          <= 0;

            arg_cnt      <= 0;

            digit_out    <= 0;

            max_val      <= 0;
            max_idx      <= 0;

            mac_prod     <= 0;
            mac_shift    <= 0;

        end else begin

            result_valid <= 1'b0;

            case (state)

            // Store pooled data
            S_FILL: begin

                if (in_valid) begin

                    for (ch = 0; ch < N_CH; ch = ch + 1)
                        input_buf[buf_pos * N_CH + ch] <=
                            $signed(data_in[ch*8 +: 8]);

                    if (buf_pos == IN_SPATIAL - 1) begin

                        buf_pos <= 0;

                        i_cnt   <= 0;
                        n_cnt   <= 0;

                        acc     <= 0;

                        state   <= S_COMPUTE;

                    end else begin
                        buf_pos <= buf_pos + 1;
                    end
                end
            end

            // Fully connected MAC
            S_COMPUTE: begin

                a_reg <= (i_cnt < IN_SIZE)
                         ? input_buf[i_cnt]
                         : 0;

                b_reg <= (i_cnt < IN_SIZE)
                         ? fc_w[n_cnt * IN_SIZE + i_cnt]
                         : 0;

                (* use_dsp = "yes" *)
                mac_prod  <= a_reg * b_reg;

                mac_shift <=
                    {{(ACC_W-16){mac_prod[15]}}, mac_prod} >>> 3;

                acc   <= acc + mac_shift;

                i_cnt <= i_cnt + 1;

                // Finished one neuron
                if (i_cnt == IN_SIZE - 1 + 3) begin

                    logits[n_cnt] <= acc + mac_shift;

                    acc   <= 0;
                    i_cnt <= 0;

                    // Finished all neurons
                    if (n_cnt == OUT_SIZE - 1) begin

                        max_val <= acc + mac_shift;
                        max_idx <= n_cnt[3:0];

                        arg_cnt <= 0;

                        n_cnt   <= 0;

                        state   <= S_ARGMAX;

                    end else begin
                        n_cnt <= n_cnt + 1;
                    end
                end
            end

            // Find maximum logit
            S_ARGMAX: begin

                if ($signed(logits[arg_cnt]) > $signed(max_val)) begin

                    max_val <= logits[arg_cnt];

                    max_idx <= arg_cnt[3:0];
                end

                if (arg_cnt == OUT_SIZE - 1) begin

                    digit_out    <= max_idx;

                    result_valid <= 1'b1;

                    // Ready for next image
                    buf_pos <= 0;

                    state <= S_FILL;

                end else begin
                    arg_cnt <= arg_cnt + 1;
                end
            end

            default: state <= S_FILL;

            endcase
        end
    end

endmodule
