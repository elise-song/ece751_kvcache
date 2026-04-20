`default_nettype none

module softmax_layer #(
    parameter int VECTOR_LEN     = 4096,
    parameter int DATA_WIDTH     = 16,
    parameter int FRAC_BITS      = 8,
    parameter int EXP_WIDTH      = 16,
    parameter int OUT_WIDTH      = 16,
    parameter int OUT_FRAC_BITS  = 16,
    parameter int LUT_ADDR_WIDTH = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_last,
    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [OUT_WIDTH-1:0]  out_data,
    output logic                  out_last,
    output logic                  busy,
    output logic                  frame_done
);

    localparam int ADDR_WIDTH = (VECTOR_LEN <= 2) ? 1 : $clog2(VECTOR_LEN);
    localparam int SUM_WIDTH  = EXP_WIDTH + ADDR_WIDTH + 2;
    localparam int unsigned VECTOR_LEN_M1_I = VECTOR_LEN - 1;
    localparam logic signed [DATA_WIDTH-1:0] SCORE_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    typedef enum logic [1:0] {
        S_LOAD = 2'd0,
        S_EXP  = 2'd1,
        S_OUT  = 2'd2
    } state_t;

    function automatic logic signed [DATA_WIDTH-1:0] fp16_to_q8_8(input logic [15:0] bits);
        logic sign_bit;
        logic [4:0] exp_field;
        logic [9:0] frac_field;
        int shift;
        int signed magnitude;
        int signed mantissa;
        int rshift;
        int signed exp_field_i;
        begin
            sign_bit   = bits[15];
            exp_field  = bits[14:10];
            frac_field = bits[9:0];
            exp_field_i = {27'd0, exp_field};

            if (exp_field == 5'h1f) begin
                fp16_to_q8_8 = sign_bit ? SCORE_MIN : {1'b0, {(DATA_WIDTH-1){1'b1}}};
            end else if ((exp_field == 5'h00) && (frac_field == 10'h000)) begin
                fp16_to_q8_8 = '0;
            end else begin
                if (exp_field == 5'h00) begin
                    mantissa = {22'd0, frac_field};
                    shift    = 1 - 15 - 10 + FRAC_BITS;
                end else begin
                    mantissa = {21'd0, 1'b1, frac_field};
                    shift    = exp_field_i - 15 - 10 + FRAC_BITS;
                end

                if (shift >= 0) begin
                    magnitude = mantissa <<< shift;
                end else begin
                    rshift = -shift;
                    magnitude = (mantissa + (1 <<< (rshift - 1))) >>> rshift;
                end

                if (sign_bit) begin
                    magnitude = -magnitude;
                end

                if (magnitude > ((1 <<< (DATA_WIDTH - 1)) - 1)) begin
                    fp16_to_q8_8 = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                end else if (magnitude < -(1 <<< (DATA_WIDTH - 1))) begin
                    fp16_to_q8_8 = SCORE_MIN;
                end else begin
                    fp16_to_q8_8 = magnitude[DATA_WIDTH-1:0];
                end
            end
        end
    endfunction

    function automatic logic [15:0] q0_16_to_fp16(input logic [OUT_WIDTH-1:0] q_value);
        int msb_idx;
        int exp_field;
        int remainder;
        int frac_field;
        int shift;
        int q_value_i;
        begin
            q_value_i = {16'd0, q_value};
            if (q_value == '0) begin
                q0_16_to_fp16 = 16'h0000;
            end else if (q_value < 16'd4) begin
                q0_16_to_fp16 = {1'b0, 5'h00, q_value[1:0], 8'h00};
            end else begin
                msb_idx = 0;
                for (int bit_idx = OUT_WIDTH - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
                    if ((msb_idx == 0) && q_value[bit_idx]) begin
                        msb_idx = bit_idx;
                    end
                end

                exp_field = msb_idx - 1;
                remainder = q_value_i - (32'd1 <<< msb_idx);

                if (msb_idx > 10) begin
                    shift = msb_idx - 10;
                    frac_field = (remainder + (1 <<< (shift - 1))) >>> shift;
                end else if (msb_idx < 10) begin
                    frac_field = remainder <<< (10 - msb_idx);
                end else begin
                    frac_field = remainder;
                end

                if (frac_field >= (1 <<< 10)) begin
                    frac_field = 0;
                    exp_field  = exp_field + 1;
                end

                if (exp_field > 30) begin
                    q0_16_to_fp16 = 16'h7bff;
                end else begin
                    q0_16_to_fp16 = {1'b0, exp_field[4:0], frac_field[9:0]};
                end
            end
        end
    endfunction

    state_t state_q, state_d;

    logic signed [DATA_WIDTH-1:0] score_mem [0:VECTOR_LEN-1];
    logic        [EXP_WIDTH-1:0]  exp_mem   [0:VECTOR_LEN-1];
    logic        [EXP_WIDTH-1:0]  exp_lut   [0:(1 << LUT_ADDR_WIDTH)-1];

    logic [ADDR_WIDTH:0] frame_len_q, frame_len_d;
    logic [ADDR_WIDTH:0] load_count_q, load_count_d;
    logic [ADDR_WIDTH:0] exp_idx_q, exp_idx_d;
    logic [ADDR_WIDTH:0] out_idx_q, out_idx_d;

    logic signed [DATA_WIDTH-1:0] max_score_q, max_score_d;
    logic [SUM_WIDTH-1:0]         sum_exp_q, sum_exp_d;

    logic do_load;
    logic do_emit;

    logic signed [DATA_WIDTH-1:0] decoded_in_score;
    logic signed [DATA_WIDTH-1:0] score_diff;
    logic [LUT_ADDR_WIDTH-1:0]    lut_index;
    logic [EXP_WIDTH-1:0]         exp_value;
    logic [ADDR_WIDTH-1:0]        load_addr;
    logic [ADDR_WIDTH-1:0]        exp_addr;
    logic [ADDR_WIDTH-1:0]        out_addr;

    logic [SUM_WIDTH+OUT_FRAC_BITS-1:0] scaled_exp;
    logic [SUM_WIDTH+OUT_FRAC_BITS-1:0] sum_exp_ext;
    logic [SUM_WIDTH+OUT_FRAC_BITS-1:0] norm_value_full;

    assign load_addr         = load_count_q[ADDR_WIDTH-1:0];
    assign exp_addr          = exp_idx_q[ADDR_WIDTH-1:0];
    assign out_addr          = out_idx_q[ADDR_WIDTH-1:0];
    assign sum_exp_ext       = {{OUT_FRAC_BITS{1'b0}}, sum_exp_q};
    assign decoded_in_score  = fp16_to_q8_8(in_data);

    initial begin : init_lut
        $readmemh("exp_lut.mem", exp_lut);
    end

    assign do_load = in_valid && in_ready;
    assign do_emit = out_valid && out_ready;

    always_comb begin
        in_ready    = (state_q == S_LOAD);
        out_valid   = (state_q == S_OUT);
        out_last    = (state_q == S_OUT) && (out_idx_q == frame_len_q - 1) && (frame_len_q != 0);
        busy        = (state_q != S_LOAD) || (load_count_q != 0);
        frame_done  = 1'b0;

        state_d      = state_q;
        frame_len_d  = frame_len_q;
        load_count_d = load_count_q;
        exp_idx_d    = exp_idx_q;
        out_idx_d    = out_idx_q;
        max_score_d  = max_score_q;
        sum_exp_d    = sum_exp_q;

        if (do_load) begin
            if (load_count_q == 0 || decoded_in_score > max_score_q) begin
                max_score_d = decoded_in_score;
            end

            if (in_last || (load_count_q == VECTOR_LEN_M1_I[ADDR_WIDTH:0])) begin
                frame_len_d  = load_count_q + 1;
                load_count_d = '0;
                exp_idx_d    = '0;
                out_idx_d    = '0;
                sum_exp_d    = '0;
                state_d      = S_EXP;
            end else begin
                load_count_d = load_count_q + 1;
            end
        end

        if (state_q == S_EXP) begin
            if (exp_idx_q == frame_len_q - 1) begin
                out_idx_d = '0;
                state_d   = S_OUT;
            end else begin
                exp_idx_d = exp_idx_q + 1;
            end
        end

        if (do_emit) begin
            if (out_idx_q == frame_len_q - 1) begin
                state_d      = S_LOAD;
                frame_len_d  = '0;
                load_count_d = '0;
                exp_idx_d    = '0;
                out_idx_d    = '0;
                max_score_d  = SCORE_MIN;
                sum_exp_d    = '0;
                frame_done   = 1'b1;
            end else begin
                out_idx_d = out_idx_q + 1;
            end
        end
    end

    always_comb begin
        if (score_mem[exp_addr] > max_score_q) begin
            score_diff = '0;
        end else begin
            score_diff = max_score_q - score_mem[exp_addr];
        end

        if (score_diff >= (8 <<< FRAC_BITS)) begin
            lut_index = {LUT_ADDR_WIDTH{1'b1}};
        end else begin
            lut_index = score_diff[3 +: LUT_ADDR_WIDTH];
        end

        exp_value = exp_lut[lut_index];
    end

    always_comb begin
        if (sum_exp_q == 0) begin
            scaled_exp      = '0;
            norm_value_full = '0;
            out_data        = 16'h0000;
        end else begin
            scaled_exp      = {{(SUM_WIDTH + OUT_FRAC_BITS - EXP_WIDTH){1'b0}}, exp_mem[out_addr]} << OUT_FRAC_BITS;
            norm_value_full = (scaled_exp + (sum_exp_ext >> 1)) / sum_exp_ext;
            if (|norm_value_full[SUM_WIDTH+OUT_FRAC_BITS-1:OUT_WIDTH]) begin
                out_data = q0_16_to_fp16({OUT_WIDTH{1'b1}});
            end else begin
                out_data = q0_16_to_fp16(norm_value_full[OUT_WIDTH-1:0]);
            end
        end
    end

    integer idx;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q      <= S_LOAD;
            frame_len_q  <= '0;
            load_count_q <= '0;
            exp_idx_q    <= '0;
            out_idx_q    <= '0;
            max_score_q  <= SCORE_MIN;
            sum_exp_q    <= '0;

            for (idx = 0; idx < VECTOR_LEN; idx = idx + 1) begin
                score_mem[idx] <= '0;
                exp_mem[idx]   <= '0;
            end
        end else begin
            state_q      <= state_d;
            frame_len_q  <= frame_len_d;
            load_count_q <= load_count_d;
            exp_idx_q    <= exp_idx_d;
            out_idx_q    <= out_idx_d;
            max_score_q  <= max_score_d;
            sum_exp_q    <= sum_exp_d;

            if (do_load) begin
                score_mem[load_addr] <= decoded_in_score;
            end

            if (state_q == S_EXP) begin
                exp_mem[exp_addr] <= exp_value;
                sum_exp_q         <= sum_exp_q + {{(SUM_WIDTH - EXP_WIDTH){1'b0}}, exp_value};
            end
        end
    end

endmodule

`default_nettype wire
