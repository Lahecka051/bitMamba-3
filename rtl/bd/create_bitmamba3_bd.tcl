################################################################################
# create_bitmamba3_bd.tcl — Vivado block design for Zybo Z7-20 BitMamba-3 demo.
#
# Generates a Vivado IPI block diagram that wires:
#   - Zynq-7000 PS (axi_dot_hp baseline reused: PS-side AXI HP0 + AXI GP0)
#   - top_bitmamba3_block.v as a Verilog wrapper instantiated through the
#     Module Reference flow (Add Module to Block Design)
#   - DDR3 controller (PS-side) reachable via AXI HP0 for weight + activation
#     streaming
#   - AXI Lite interconnect from GP0 to top block control registers
#
# Usage:
#   vivado -mode tcl -source rtl/bd/create_bitmamba3_bd.tcl
#
# Output:
#   rtl/vivado_project/gguf_bitmamba3_proj/  (Vivado project tree)
#   rtl/vivado_project/gguf_bitmamba3_proj/gguf_bitmamba3_proj.runs/synth_1
################################################################################

set proj_name "gguf_bitmamba3_proj"
set proj_dir  "[file normalize "[file dirname [info script]]/../vivado_project/$proj_name"]"
set rtl_dir   "[file normalize "[file dirname [info script]]/../cores/bitmamba3"]"

# Zybo Z7-20 part
set part "xc7z020clg400-1"
set board_part "digilentinc.com:zybo-z7-20:part0:1.0"

create_project -force $proj_name $proj_dir -part $part
set_property board_part $board_part [current_project]

# Add RTL sources
add_files -norecurse -fileset sources_1 [glob -directory $rtl_dir *.v]
update_compile_order -fileset sources_1

# Create block design
create_bd_design "design_1"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config { \
    make_external "FIXED_IO, DDR" \
    apply_board_preset "1" \
    Master "Disable" \
    Slave "Disable" \
} [get_bd_cells ps7]

# Enable AXI HP0 (high-throughput DDR access for weights / activations)
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_FCLK_CLK0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7]

# Add the BitMamba-3 block as Module Reference
create_bd_cell -type module -reference top_bitmamba3_block bm3_top

# Auto-connect AXI Lite (PS GP0 -> bm3_top s_axi_lite) and clocks/resets
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Master "/ps7/M_AXI_GP0" \
    Slave  "/bm3_top/s_axi_lite" \
    intc_ip "New AXI Interconnect" \
    Clk_xbar "Auto" Clk_master "Auto" Clk_slave "Auto" \
} [get_bd_intf_pins bm3_top/s_axi_lite] 2> /dev/null
# Note: Module Reference s_axi_lite ports may need manual connection if the
# above automation rule does not match the inferred interface. See the
# top_bitmamba3_block port list for explicit signal-by-signal connection.

# Clock and reset
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]      [get_bd_pins bm3_top/ACLK]
# Use Processor System Reset to derive resetn (active-low)
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_clk
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]      [get_bd_pins rst_clk/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N]  [get_bd_pins rst_clk/ext_reset_in]
connect_bd_net [get_bd_pins rst_clk/peripheral_aresetn] [get_bd_pins bm3_top/ARESETN]

# Validate, save, generate wrapper
validate_bd_design
save_bd_design

make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [get_property DIRECTORY [current_fileset]]/design_1_wrapper.v
update_compile_order -fileset sources_1

# Synthesis run
launch_runs synth_1 -jobs 4
wait_on_run synth_1

puts "Synthesis complete."
report_utilization -file [file join $proj_dir "utilization_synth.rpt"]
puts "Utilization report saved to utilization_synth.rpt"
