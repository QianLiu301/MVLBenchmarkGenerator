// Example submission in SystemC: a ternary ALU with 8 digits (benchmark alu_k3_8t).
// The numbers are 0 .. 3^8 - 1 = 6560, held in 13 bits, and every operation
// wraps around modulo 3^8 = 6561.
#include <systemc.h>
#include <cstdio>

SC_MODULE(mvl_alu_3_8bit) {
    sc_in<bool>             clk;
    sc_in<bool>             rst;
    sc_in<sc_uint<13> >     a;
    sc_in<sc_uint<13> >     b;
    sc_in<sc_uint<4> >      opcode;    // 0 ADD, 1 SUB, 2 MUL, 3 NEG, 4 INC, 5 DEC
    sc_out<sc_uint<13> >    result;
    sc_out<bool>            zero;      // result is 0
    sc_out<bool>            negative;  // result >= 3280, the upper half of the range
    sc_out<bool>            carry;     // ADD wrapped, SUB borrowed, INC/DEC wrapped

    static const unsigned MOD = 6561, HALF = 3280;

    void step() {
        if (rst.read()) {
            result.write(0); zero.write(true); negative.write(false); carry.write(false);
            return;
        }
        unsigned long x = a.read().to_uint(), y = b.read().to_uint(), r = 0;
        bool c = false;
        switch (opcode.read().to_uint()) {
            case 0: r = x + y;       c = (r >= MOD);      break;  // ADD
            case 1: r = x + MOD - y; c = (x < y);         break;  // SUB
            case 2: r = x * y;                            break;  // MUL
            case 3: r = MOD - x;                          break;  // NEG
            case 4: r = x + 1;       c = (x == MOD - 1);  break;  // INC
            case 5: r = x + MOD - 1; c = (x == 0);        break;  // DEC
            default: r = 0;
        }
        r %= MOD;
        result.write(r);
        zero.write(r == 0);
        negative.write(r >= HALF);
        carry.write(c);
    }

    SC_CTOR(mvl_alu_3_8bit) {
        SC_METHOD(step);
        sensitive << clk.pos();
    }
};

// The file's own tests: one line per test, in the format the library reads.
// (The library replaces this sc_main with its own when it injects its vectors.)
int sc_main(int argc, char* argv[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst, zero, negative, carry;
    sc_signal<sc_uint<13> > a, b, result;
    sc_signal<sc_uint<4> > opcode;

    mvl_alu_3_8bit dut("dut");
    dut.clk(clk); dut.rst(rst); dut.a(a); dut.b(b); dut.opcode(opcode);
    dut.result(result); dut.zero(zero); dut.negative(negative); dut.carry(carry);

    rst.write(true);
    sc_start(20, SC_NS);
    rst.write(false);

    const unsigned as[] = {0, 6560, 1, 100}, bs[] = {0, 6560, 2, 6500};
    const char* names[] = {"ADD", "SUB", "MUL", "NEG", "INC", "DEC"};
    int n = 0;
    for (int op = 0; op < 6; op++)
        for (int i = 0; i < 4; i++) {
            a.write(as[i]); b.write(bs[i]); opcode.write(op);
            sc_start(20, SC_NS);              // two clock periods: one rising edge after the change
            printf("Test %d: %s A=%u B=%u -> R=%u Z=%d N=%d C=%d\n", ++n, names[op], as[i], bs[i],
                   (unsigned)result.read().to_uint(), (int)zero.read(), (int)negative.read(),
                   (int)carry.read());
        }
    return 0;
}
