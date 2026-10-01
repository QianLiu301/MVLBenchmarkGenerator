// Example submission: a ternary ALU with 8 digits (benchmark alu_k3_8t).
// The numbers are 0 .. 3^8 - 1 = 6560, held in 13 bits, and every operation
// wraps around modulo 3^8 = 6561.

module mvl_alu_3_8bit (
    input  wire        clk,
    input  wire        rst,
    input  wire [12:0] a,
    input  wire [12:0] b,
    input  wire [3:0]  opcode,    // 0 ADD, 1 SUB, 2 MUL, 3 NEG, 4 INC, 5 DEC
    output reg  [12:0] result,
    output reg         zero,      // result is 0
    output reg         negative,  // result >= 3280, the upper half of the range
    output reg         carry      // ADD wrapped, SUB borrowed, INC/DEC wrapped
);
    localparam [25:0] MOD  = 26'd6561;
    localparam [25:0] HALF = 26'd3280;

    reg [25:0] r;   // wide enough for a product
    reg        c;

    always @(posedge clk) begin
        if (rst) begin
            result <= 13'd0; zero <= 1'b1; negative <= 1'b0; carry <= 1'b0;
        end else begin
            c = 1'b0;
            case (opcode)
                4'd0: begin r = a + b;       c = (r >= MOD);       end  // ADD
                4'd1: begin r = a + MOD - b; c = (a < b);          end  // SUB
                4'd2: begin r = a * b;                             end  // MUL
                4'd3: begin r = MOD - a;                           end  // NEG
                4'd4: begin r = a + 1;       c = (a == MOD - 1);   end  // INC
                4'd5: begin r = a + MOD - 1; c = (a == 0);         end  // DEC
                default: r = 0;
            endcase
            r = r % MOD;
            result   <= r[12:0];
            zero     <= (r == 0);
            negative <= (r >= HALF);
            carry    <= c;
        end
    end
endmodule

// The file's own tests: one line per test, in the format the library reads.
// (The library replaces this module with its own when it injects its vectors.)
module mvl_alu_3_8bit_tb;
    reg         clk = 1'b0, rst = 1'b1;
    reg  [12:0] a = 0, b = 0;
    reg  [3:0]  opcode = 0;
    wire [12:0] result;
    wire        zero, negative, carry;

    reg  [12:0] as [0:3];
    reg  [12:0] bs [0:3];
    reg  [23:0] name;          // three characters
    integer     op, i, n;

    mvl_alu_3_8bit dut (.clk(clk), .rst(rst), .a(a), .b(b), .opcode(opcode),
                        .result(result), .zero(zero), .negative(negative), .carry(carry));

    always #5 clk = ~clk;

    initial begin
        as[0] = 0;    bs[0] = 0;
        as[1] = 6560; bs[1] = 6560;
        as[2] = 1;    bs[2] = 2;
        as[3] = 100;  bs[3] = 6500;
        #20 rst = 1'b0;
        n = 0;
        for (op = 0; op < 6; op = op + 1)
            for (i = 0; i < 4; i = i + 1) begin
                a = as[i]; b = bs[i]; opcode = op;
                @(posedge clk); @(posedge clk); #1;   // result is ready after one edge
                case (op)
                    0: name = "ADD";  1: name = "SUB";  2: name = "MUL";
                    3: name = "NEG";  4: name = "INC";  default: name = "DEC";
                endcase
                n = n + 1;
                $display("Test %0d: %s A=%0d B=%0d -> R=%0d Z=%0d N=%0d C=%0d",
                         n, name, a, b, result, zero, negative, carry);
            end
        $finish;
    end
endmodule
