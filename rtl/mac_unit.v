module mac_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clr,
    input  wire        en,
    input  wire signed [7:0]  pixel,
    input  wire signed [7:0]  weight,
    output reg  signed [19:0] acc
);
    wire signed [15:0] product      = pixel * weight;
    wire signed [19:0] prod_shifted = {{4{product[15]}}, product} >>> 3;

    always @(posedge clk) begin
        if (!rst_n || clr)
            acc <= 20'sd0;
        else if (en)
            acc <= acc + prod_shifted;
    end
endmodule
