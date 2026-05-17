`timescale 1ns/1ps
module mnist_cnn_tb;

    reg        clk   = 0;
    reg        rst_n = 0;
    reg        pixel_valid = 0;
    reg  [7:0] pixel_raw   = 0;
    wire [3:0] digit_out;
    wire       result_valid;

    always #5 clk = ~clk;

    mnist_cnn_top dut (
        .clk(clk), .rst_n(rst_n),
        .pixel_valid(pixel_valid), .pixel_raw(pixel_raw),
        .digit_out(digit_out), .result_valid(result_valid)
    );

    reg [7:0] test_image [0:783];
    initial $readmemh("digit8_test.hex", test_image);

    integer k;
    initial begin
        $dumpfile("sim.vcd"); $dumpvars(0, mnist_cnn_tb);
        #20 rst_n = 1; #10;

        $display("[TB] Streaming 784 pixels...");
        for (k = 0; k < 784; k = k+1) begin
            @(posedge clk);
            pixel_valid <= 1;
            pixel_raw   <= test_image[k];
        end
        @(posedge clk); pixel_valid <= 0;

        $display("[TB] Stream done. Waiting for result_valid...");
        wait(result_valid == 1);
        @(posedge clk);
        $display("[TB] digit_out = %0d", digit_out);
        $finish;
    end

    // Monitor key pipeline signals
    always @(posedge clk) begin
        if (dut.u_conv1.out_valid)
            $display("[MON] t=%0t conv1_valid, conv1_out[0]=%0d",
                     $time, dut.u_conv1.conv_out[7:0]);
        if (dut.u_pool1.out_valid)
            $display("[MON] t=%0t pool1_valid, pool1_out[0]=%0d",
                     $time, dut.u_pool1.data_out[7:0]);
        if (dut.u_conv2.out_valid)
            $display("[MON] t=%0t conv2_valid",  $time);
        if (dut.u_pool2.out_valid)
            $display("[MON] t=%0t pool2_valid",  $time);
        if (result_valid)
            $display("[MON] t=%0t RESULT digit=%0d", $time, digit_out);
    end

    initial #50000000 begin
        $display("[TB] TIMEOUT"); $finish;
    end
endmodule
