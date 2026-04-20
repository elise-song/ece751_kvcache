`default_nettype none

module tb_softmax_layer;
    parameter int VECTOR_LEN = 8;
    parameter int NUM_CASES  = 3;

    logic clk;
    logic rst_n;

    logic        in_valid;
    logic        in_ready;
    logic [15:0] in_data;
    logic        in_last;

    logic         out_valid;
    logic         out_ready;
    logic [15:0]  out_data;
    logic         out_last;
    logic         busy;
    logic         frame_done;

    logic [15:0] input_mem    [0:(NUM_CASES * VECTOR_LEN)-1];
    logic [15:0] expected_mem [0:(NUM_CASES * VECTOR_LEN)-1];

    int case_idx;
    int elem_idx;
    int out_idx;
    /* verilator lint_off UNUSEDSIGNAL */
    integer flat_idx;
    /* verilator lint_on UNUSEDSIGNAL */

    softmax_layer #(
        .VECTOR_LEN(VECTOR_LEN),
        .DATA_WIDTH(16),
        .FRAC_BITS(8),
        .EXP_WIDTH(16),
        .OUT_WIDTH(16),
        .OUT_FRAC_BITS(16),
        .LUT_ADDR_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .in_last(in_last),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_data(out_data),
        .out_last(out_last),
        .busy(busy),
        .frame_done(frame_done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task automatic drive_frame(input int frame_id);
        begin
            for (elem_idx = 0; elem_idx < VECTOR_LEN; elem_idx = elem_idx + 1) begin
                flat_idx = frame_id * VECTOR_LEN + elem_idx;

                do begin
                    @(posedge clk);
                end while (!in_ready);

                in_valid = 1'b1;
                in_data  = input_mem[flat_idx];
                in_last  = (elem_idx == VECTOR_LEN - 1);
            end

            @(posedge clk);
            in_valid = 1'b0;
            in_data  = '0;
            in_last  = 1'b0;
        end
    endtask

    task automatic check_frame(input int frame_id);
        begin
            out_idx = 0;
            while (out_idx < VECTOR_LEN) begin
                @(posedge clk);

                if (out_valid) begin
                    flat_idx = frame_id * VECTOR_LEN + out_idx;
                    if (out_data !== expected_mem[flat_idx]) begin
                        $display(
                            "Mismatch in case %0d output %0d: got 0x%04h expected 0x%04h",
                            frame_id,
                            out_idx,
                            out_data,
                            expected_mem[flat_idx]
                        );
                        $fatal(1);
                    end

                    if ((out_idx == VECTOR_LEN - 1) && !out_last) begin
                        $display("Expected out_last on final element of case %0d", frame_id);
                        $fatal(1);
                    end

                    if ((out_idx != VECTOR_LEN - 1) && out_last) begin
                        $display("Unexpected out_last early on case %0d output %0d", frame_id, out_idx);
                        $fatal(1);
                    end

                    out_idx = out_idx + 1;
                end
            end
        end
    endtask

    initial begin
        $readmemh("tb_inputs.mem", input_mem);
        $readmemh("tb_expected.mem", expected_mem);

        rst_n     = 1'b0;
        in_valid  = 1'b0;
        in_data   = '0;
        in_last   = 1'b0;
        out_ready = 1'b1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (case_idx = 0; case_idx < NUM_CASES; case_idx = case_idx + 1) begin
            drive_frame(case_idx);
            check_frame(case_idx);
            repeat (2) @(posedge clk);
        end

        if (busy || frame_done) begin
            $display("Final status busy=%0b frame_done=%0b", busy, frame_done);
        end

        $display("All softmax_layer tests passed.");
        $finish;
    end

endmodule

`default_nettype wire
