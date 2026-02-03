import os
from imgui_bundle import imgui
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
import sys
sys.path.append("./gaussian-splatting")
from widgets.widget import Widget
import torch


# Evaluation methods configuration - EVS Splitting Methods
EVAL_METHODS = [
    ('Hard_Clipping', {
        'clip_model': True, 'enable_evs': False
    }),
    # Naive EVS (split all intersecting Gaussians) - 1 to 5 passes
    ('EVS_Naive_1pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 1, 'evs_split_mode': 0
    }),
    ('EVS_Naive_2pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 2, 'evs_split_mode': 0
    }),
    ('EVS_Naive_3pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 3, 'evs_split_mode': 0
    }),
    ('EVS_Naive_4pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 4, 'evs_split_mode': 0
    }),
    ('EVS_Naive_5pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 5, 'evs_split_mode': 0
    }),
    # Proxy control with asymmetry cost: 1-min(Cl,Cr) - 1 to 5 passes
    ('EVS_Proxy_Asym_1pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 1, 'evs_split_mode': 1, 'evs_cost_mode': 0
    }),
    ('EVS_Proxy_Asym_2pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 2, 'evs_split_mode': 1, 'evs_cost_mode': 0
    }),
    ('EVS_Proxy_Asym_3pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 3, 'evs_split_mode': 1, 'evs_cost_mode': 0
    }),
    ('EVS_Proxy_Asym_4pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 4, 'evs_split_mode': 1, 'evs_cost_mode': 0
    }),
    ('EVS_Proxy_Asym_5pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 5, 'evs_split_mode': 1, 'evs_cost_mode': 0
    }),
    # Proxy control with conservative cost: |Cl-Cr| - 1 to 5 passes
    ('EVS_Proxy_Cons_1pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 1, 'evs_split_mode': 1, 'evs_cost_mode': 1
    }),
    ('EVS_Proxy_Cons_2pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 2, 'evs_split_mode': 1, 'evs_cost_mode': 1
    }),
    ('EVS_Proxy_Cons_3pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 3, 'evs_split_mode': 1, 'evs_cost_mode': 1
    }),
    ('EVS_Proxy_Cons_4pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 4, 'evs_split_mode': 1, 'evs_cost_mode': 1
    }),
    ('EVS_Proxy_Cons_5pass', {
        'clip_model': True, 'enable_evs': True,
        'evs_max_passes': 5, 'evs_split_mode': 1, 'evs_cost_mode': 1
    }),
]


class EVSWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "EVS Splitting")

        # EVS Splitting settings
        self.enable_evs = False
        self.evs_debug = False
        self.clip_model = False

        # Adaptive EVS settings
        self.evs_max_passes = 2  # Maximum number of EVS splitting passes
        self.evs_min_split_threshold = 0  # Early termination threshold

        # Benefit-cost split control settings
        self.evs_split_mode = 0  # 0=naive (split all), 1=proxy_control (benefit-cost)
        self.evs_cost_mode = 0   # 0=1-min(Cl,Cr), 1=|Cl-Cr|
        self.evs_lambda = 1.0    # benefit-cost threshold

        # Memory optimization settings
        self.evs_mode = 0  # 0=naive, 1=scenegraph, 2=cpu_offload
        self.evs_measure_memory = False  # measure peak memory usage
        self.memory_stats = {'naive': None, 'scenegraph': None, 'cpu_offload': None}  # store memory measurements
        self.peak_stats = {'naive': None, 'scenegraph': None, 'cpu_offload': None}  # store peak memory measurements
        self.fps_stats = {'naive': None, 'scenegraph': None, 'cpu_offload': None}  # store FPS measurements

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if not show:
            return

        imgui.separator()
        label("======================================================= EV Splitting Settings =======================================================")
        imgui.separator()

        # enable EVS split
        _, self.enable_evs = imgui.checkbox("Enable EVS Split", self.enable_evs)

        # Adaptive EVS controls (only show when EVS is enabled)
        if self.enable_evs:
            _, self.evs_max_passes = imgui.slider_int("EVS Passes", self.evs_max_passes, 1, 5)
            _, self.evs_min_split_threshold = imgui.input_int("Min Split Threshold", self.evs_min_split_threshold)
            self.evs_min_split_threshold = max(0, self.evs_min_split_threshold)

            # Split mode selection (Naive vs Proxy Control)
            mode_names = ["Naive", "Proxy Control"]
            _, self.evs_split_mode = imgui.combo("Split Mode", self.evs_split_mode, mode_names)

            # Cost mode and Lambda (only show when proxy control enabled)
            if self.evs_split_mode == 1:
                cost_names = ["1-min(Cl,Cr)", "|Cl-Cr|"]
                _, self.evs_cost_mode = imgui.combo("Cost Formula", self.evs_cost_mode, cost_names)
                _, self.evs_lambda = imgui.slider_float("Lambda", self.evs_lambda, 0.0, 10.0)

            # Memory optimization mode (Naive vs SceneGraph vs CPU-Offload)
            imgui.separator()
            mem_mode_names = ["Naive (Clone)", "SceneGraph (Efficient)", "CPU-Offload (Max Save)"]
            _, self.evs_mode = imgui.combo("Memory Mode", self.evs_mode, mem_mode_names)
            _, self.evs_measure_memory = imgui.checkbox("Measure Memory", self.evs_measure_memory)

            # Update memory stats from render result (using precise gs_memory_mb)
            bytes_per_gs = None
            if hasattr(self.viz, 'result') and 'evs_stats' in self.viz.result:
                evs_stats = self.viz.result['evs_stats']
                bytes_per_gs = evs_stats.get('bytes_per_gs')
                if self.evs_measure_memory:
                    mode = evs_stats.get('evs_mode', 'naive')
                    if 'gs_memory_mb' in evs_stats:
                        self.memory_stats[mode] = evs_stats['gs_memory_mb']
                    if 'peak_memory_mb' in evs_stats:
                        self.peak_stats[mode] = evs_stats['peak_memory_mb']
                    if 'fps' in evs_stats:
                        self.fps_stats[mode] = evs_stats['fps']

            # Display bytes per Gaussian
            if bytes_per_gs is not None:
                imgui.same_line()
                imgui.text(f"  ({bytes_per_gs}B/GS)")

            # Helper function to format memory with appropriate unit
            def format_memory(mb):
                if mb is None:
                    return "--"
                bytes_val = mb * 1024 * 1024
                if bytes_val < 1024:  # < 1KB, show bytes
                    return f"{bytes_val:.0f}B"
                elif bytes_val < 1024 * 1024:  # < 1MB, show KB
                    return f"{bytes_val / 1024:.2f}KB"
                else:  # >= 1MB, show MB
                    return f"{mb:.2f}MB"

            # Display memory comparison (all three modes)
            if self.evs_measure_memory:
                naive_mem = self.memory_stats.get('naive')
                sg_mem = self.memory_stats.get('scenegraph')
                cpu_mem = self.memory_stats.get('cpu_offload')
                naive_peak = self.peak_stats.get('naive')
                sg_peak = self.peak_stats.get('scenegraph')
                cpu_peak = self.peak_stats.get('cpu_offload')
                naive_fps = self.fps_stats.get('naive')
                sg_fps = self.fps_stats.get('scenegraph')
                cpu_fps = self.fps_stats.get('cpu_offload')

                # Helper to format individual stat
                def fmt_stat(val, unit=""):
                    return f"{val:.1f}{unit}" if val is not None else "--"

                # Display final memory comparison
                has_any_mem = naive_mem is not None or sg_mem is not None or cpu_mem is not None
                if has_any_mem:
                    imgui.text(f"  Final: Naive {format_memory(naive_mem)} | SG {format_memory(sg_mem)} | CPU {format_memory(cpu_mem)}")
                else:
                    imgui.text(f"  (Switch modes to measure)")

                # Display peak memory comparison
                has_any_peak = naive_peak is not None or sg_peak is not None or cpu_peak is not None
                if has_any_peak:
                    imgui.text(f"  Peak:  Naive {format_memory(naive_peak)} | SG {format_memory(sg_peak)} | CPU {format_memory(cpu_peak)}")

                # Display FPS comparison
                has_any_fps = naive_fps is not None or sg_fps is not None or cpu_fps is not None
                if has_any_fps:
                    imgui.text(f"  FPS:   Naive {fmt_stat(naive_fps)} | SG {fmt_stat(sg_fps)} | CPU {fmt_stat(cpu_fps)}")

        # debug flag
        _, self.evs_debug = imgui.checkbox("Debug Render", self.evs_debug)

        # clip_model flag - when enabled, cull Gaussians on the wrong side of clipping plane
        _, self.clip_model = imgui.checkbox("Clip Model (Cull Gaussians)", self.clip_model)

        imgui.separator()
        imgui.separator()

        # output parameters to renderer
        self.viz.args.enable_evs = self.enable_evs
        self.viz.args.evs_debug = self.evs_debug
        self.viz.args.clip_model = self.clip_model
        self.viz.args.evs_max_passes = self.evs_max_passes
        self.viz.args.evs_min_split_threshold = self.evs_min_split_threshold
        self.viz.args.evs_split_mode = self.evs_split_mode
        self.viz.args.evs_cost_mode = self.evs_cost_mode
        self.viz.args.evs_lambda = self.evs_lambda
        # Memory optimization parameters
        evs_mode_map = {0: 'naive', 1: 'scenegraph', 2: 'cpu_offload'}
        self.viz.args.evs_mode = evs_mode_map.get(self.evs_mode, 'naive')
        self.viz.args.evs_measure_memory = self.evs_measure_memory

