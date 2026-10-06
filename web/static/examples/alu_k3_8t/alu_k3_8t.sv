// Example submission in SystemVerilog: a ternary ALU with 8 digits (benchmark alu_k3_8t).
// The numbers are 0 .. 3^8 - 1 = 6560, held in 13 bits, and every operation
// wraps around modulo 3^8 = 6561.

module mvl_alu_3_8bit (
    input  logic        clk,
    input  logic        rst,
    input  logic [12:0] a,
    input  logic [12:0] b,
    input  logic [3:0]  opcode,    // 0 ADD, 1 SUB, 2 MUL, 3 NEG, 4 INC, 5 DEC
    output logic [12:0] result,
    output logic        zero,      // result is 0
    output logic        negative,  // result >= 3280, the upper half of the range
    output logic        carry      // ADD wrapped, SUB borrowed, INC/DEC wrapped
);
    localparam logic [25:0] MOD  = 26'd6561;
    localparam logic [25:0] HALF = 26'd3280;

    logic [25:0] r;   // wide enough for a product
    logic        c;

    always_comb begin
        c = 1'b0;
        unique case (opcode)
            4'd0: begin r = a + b;       c = (r >= MOD);       end  // ADD
            4'd1: begin r = a + MOD - b; c = (a < b);          end  // SUB
            4'd2: begin r = a * b;                             end  // MUL
            4'd3: begin r = MOD - a;                           end  // NEG
            4'd4: begin r = a + 1;       c = (a == MOD - 1);   end  // INC
            4'd5: begin r = a + MOD - 1; c = (a == 0);         end  // DEC
            default: r = '0;
        endcase
        r = r % MOD;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            result <= '0; zero <= 1'b1; negative <= 1'b0; carry <= 1'b0;
        end else begin
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
    logic        clk = 1'b0, rst = 1'b1;
    logic [12:0] a = '0, b = '0;
    logic [3:0]  opcode = '0;
    logic [12:0] result;
    logic        zero, negative, carry;

    logic [12:0] as [4] = '{13'd0, 13'd6560, 13'd1, 13'd100};
    logic [12:0] bs [4] = '{13'd0, 13'd6560, 13'd2, 13'd6500};
    string       names [6] = '{"ADD", "SUB", "MUL", "NEG", "INC", "DEC"};
    int          n = 0;

    mvl_alu_3_8bit dut (.*);

    always #5 clk = ~clk;

    initial begin
        #20 rst = 1'b0;
        for (int op = 0; op < 6; op++)
            for (int i = 0; i < 4; i++) begin
                a = as[i]; b = bs[i]; opcode = op;
                @(posedge clk); @(posedge clk); #1;   // result is ready after one edge
                n++;
                $display("Test %0d: %s A=%0d B=%0d -> R=%0d Z=%0d N=%0d C=%0d",
                         n, names[op], a, b, result, zero, negative, carry);
            end
        $finish;
    end
endmodule
