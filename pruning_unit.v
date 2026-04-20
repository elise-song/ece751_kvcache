//  Pruning Unit Pseudocode
//  inputs: attention_score_i, new_k, new_v, last_recent_k, last_recent_v, mask reg low_index, low_score # local variables / storage elements for i in range(h2_size):
//  @clk
//  score = sum(attention_score_i) # 4096 × 1 it (score < low_score):
//  low_index = i
//  low_ score = score
//  # last_recent
//  @clk
//  last_recent_score = attention_score_i
//  if (last_recent_score > low_score):
//  return low_index, pruned(last_recent_k)
//  else:
//  return h2
//  _size + 1, null

module fp16_add (
    input  wire [15:0] a,   // positive FP16
    input  wire [15:0] b,   // positive FP16
    output reg  [15:0] result
);
    // ---- Unpack ----
    wire [4:0]  a_exp = a[14:10];
    wire [9:0]  a_man = a[9:0];
    wire [4:0]  b_exp = b[14:10];
    wire [9:0]  b_man = b[9:0];
 
    // ---- Order so that large has the bigger exponent ----
    wire        swap      = (b_exp > a_exp);
    wire [4:0]  exp_large = swap ? b_exp : a_exp;
    wire [9:0]  man_large = swap ? b_man : a_man;
    wire [4:0]  exp_small = swap ? a_exp : b_exp;
    wire [9:0]  man_small = swap ? a_man : b_man;
 
    // ---- Restore implicit leading 1 (normalised) ----
    wire [10:0] sig_large = {1'b1, man_large};   // 11 bits
    wire [10:0] sig_small = {1'b1, man_small};   // 11 bits
 
    // ---- Exponent difference (capped at 13 to avoid over-shifting) ----
    wire [4:0]  raw_diff  = exp_large - exp_small;
    wire [4:0]  diff      = (raw_diff > 5'd13) ? 5'd13 : raw_diff;
 
    // ---- Extend both significands with 3 guard bits ----
    wire [13:0] ext_large = {sig_large, 3'b000};           // 14 bits, no shift
    wire [13:0] ext_small = {sig_small, 3'b000} >> diff;   // 14 bits, shifted right
 
    // ---- Add ----
    wire [14:0] sum = {1'b0, ext_large} + {1'b0, ext_small};  // 15 bits
 
    // ---- Normalise ----
    // If bit 14 set there was a carry: shift right 1, bump exponent
    wire        carry      = sum[14];
    wire [13:0] norm_sig   = carry ? sum[14:1]  : sum[13:0];
    wire [4:0]  norm_exp   = carry ? (exp_large + 5'd1) : exp_large;
 
    // norm_sig layout: [13]=implicit 1, [12:3]=mantissa bits, [2]=round, [1:0]=sticky
    wire [9:0]  mant_trunc = norm_sig[12:3];
    wire        round_bit  = norm_sig[2];
    wire        sticky     = |norm_sig[1:0];
 
    // Round to nearest even
    wire        do_round   = round_bit & (sticky | mant_trunc[0]);
    wire [9:0]  mant_round = mant_trunc + {9'd0, do_round};
 
    // Mantissa overflow after rounding bumps the exponent
    wire        mant_ov    = (mant_round == 10'h000) & do_round;
    wire [4:0]  final_exp  = mant_ov ? (norm_exp + 5'd1) : norm_exp;
    wire [9:0]  final_mant = mant_round;
 
    always @(*) begin
        if (a == 16'h0000)
            result = b;
        else if (b == 16'h0000)
            result = a;
        else if (final_exp >= 5'd31)
            result = 16'h7C00;    // overflow to +Inf
        else
            result = {1'b0, final_exp, final_mant};
    end
endmodule
 
 
// FP16 pruning unit
module pruning_unit (
    clk,
    reset,
    num_tokens, // must be less than cache width = 1024
    previous_scores, // array of previous scores
    attention_score_i, // streaming attention score input
    last_recent_k, // last recent key vector 
    low_index,
    pruned_k,
    done
);
    parameter CACHE_WIDTH = 1024; // add 512 for recent cache for total size
    parameter NUM_CHANNELS = 4096;
    parameter PRUNED_CHANNELS = 2048; // pruning 50%
    parameter H2_SIZE = 512;

    input wire clk;
    input wire reset;
    input wire [15:0] num_tokens;
    input wire [CACHE_WIDTH-1:0][15:0] previous_scores;
    input wire [15:0] attention_score_i;
    input wire [NUM_CHANNELS-1:0][15:0] last_recent_k;

    output reg [15:0] low_index;
    output reg [PRUNED_CHANNELS-1:0][15:0] pruned_k;
    output reg done;

    reg [15:0] low_score;
    
    reg [15:0] score;
    reg [15:0] index;

    reg [1:0] STATE;
    parameter [1:0] INIT = 2'd0;
    parameter [1:0] STREAMING_INPUT = 2'd1;
    parameter [1:0] LAST_INPUT = 2'd2;
    parameter [1:0] DONE = 2'd3;
    
    always @ (posedge clk) begin

        if (reset)  STATE <= INIT;

        else begin

            case (STATE)
                INIT:               STATE <= STREAMING_INPUT;   
                STREAMING_INPUT:    if (index == num_tokens - 2) STATE <= LAST_INPUT;
                                    else STATE <= STREAMING_INPUT;
                LAST_INPUT:         STATE <= DONE;
                DONE:               STATE <= DONE;
            endcase
        end
    end

    wire [15:0] cur_score = previous_scores[index];
    wire [15:0] new_score;
 
    fp16_add u_add (
        .a      (cur_score),
        .b      (attention_score_i),
        .result (new_score)
    );

    always @(posedge clk) begin

        if (reset) begin
        end

        else begin
            case (STATE)
                
                INIT: begin  
                    low_score <= 16'hFFFF; // Initialize to max value
                    index <= 0;
                    done <= 0;
                end

                STREAMING_INPUT: begin                    
                    if (new_score < low_score) begin
                        low_index <= index;
                        low_score <= new_score;
                    end
                    index <= index + 1;

                end

                LAST_INPUT: begin
                    if (new_score < low_score) begin
                        low_index <= index;
                        low_score <= new_score; 
                        pruned_k <= last_recent_k[PRUNED_CHANNELS-1:0]; 
                    end
                    done <= 1;
                end
            endcase
        end    
    end

endmodule

module testbench;
    reg clk;
    reg reset;
    reg is_streaming;
    reg [15:0] attention_score_i;
    reg [1023:0][15:0] previous_scores;
    reg [4095:0][15:0] last_recent_k;
    wire [15:0] low_index;
    wire [2047:0][15:0] pruned_k;
    wire done;

    wire [15:0] pruned_k_0 = pruned_k[0];

    localparam FP16_0P9  = 16'h3B33;
    localparam FP16_0P8  = 16'h3A66;
    localparam FP16_0P05 = 16'h2A66;
    localparam FP16_ZERO = 16'h0000;
    localparam FP16_0P5 = 16'h3800;

    localparam num_tokens = 16'd200;
    localparam exp_low_index = 199;


    pruning_unit dut (
        .clk(clk),
        .reset(reset),
        .num_tokens(num_tokens), 
        .previous_scores(previous_scores), 
        .attention_score_i(attention_score_i),
        .last_recent_k(last_recent_k),
        .low_index(low_index),
        .pruned_k(pruned_k),
        .done(done)
    );

    initial clk = 0;
    always  #5 clk = ~clk;

    integer i;
    integer seed;
 
    initial begin
        // set previous_scores[0-198] = ~0.9, previous_scores[199] = ~0.8, previous_scores[200-1023] = 0
        for (i = 0; i < num_tokens; i = i + 1)
            if (i == exp_low_index) 
                previous_scores[i] = FP16_0P8; // set one score to 0.05 to test pruning
            else
                previous_scores[i] = FP16_0P9;
  
        for (i = num_tokens; i < 1024; i = i + 1)
            previous_scores[i] = FP16_ZERO;
 
        //  set last_recent_k random FP16 values in (0, 1)
        seed = 42;
        for (i = 0; i < 4096; i = i + 1) begin
            last_recent_k[i] = FP16_0P5;
            seed = seed + 1;
        end
 
        // Reset for two clock cycles
        reset             = 1;
        attention_score_i = FP16_ZERO;
 
        @(posedge clk); #1;
        @(posedge clk); #1;
        reset = 0;
 
        
        //  Stream 200 cycles with attention_score_i = FP16(0.05)
        //    
        for (i = 0; i < num_tokens; i = i + 1) begin
            attention_score_i = FP16_0P05;
            @(posedge clk); #1;
        end
 
        
        //  Wait for DONE then report results
        
        @(posedge clk); #1;
        @(posedge clk); #1;
 
        $display("===========================================");
        $display("Pruning unit simulation complete.");
        $display("  low_index      = %0d  (expected: %0d)", low_index, exp_low_index);
        $display("  pruned_k_0     = %0d  (expected: %0d)", pruned_k_0, exp_low_index == num_tokens-1 ? FP16_0P5 : 0); // pruned_k should be last_recent_k if low_index is 199, else 0
        $display("===========================================");
 
        $finish;
    end

    initial begin
        $dumpfile("pruning_unit.vcd");
        $dumpvars(0, testbench);
    end

endmodule

// iverilog -g2012 -o pruning_unit pruning_unit.v
// vvp pruning_unit
// surfer pruning_unit.vcd 