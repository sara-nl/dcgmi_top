import argparse
import copy
import curses
import io
import locale
import logging
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import redirect_stdout
import datetime
from functools import partial

import plotext as plt

from dcgmi_top.dcgm_reader import DCGMReader, GPUStats

locale.setlocale(locale.LC_ALL, '')
code = locale.getpreferredencoding()
COLUMNS = ["gpu_utilization", "mem_copy_utilization", "gr_engine_active", "sm_active", "sm_occupancy", "tensor_active",
           "dram_active", "pcie_tx_bytes", "pcie_rx_bytes", "nvlink_tx_bytes",
           "nvlink_rx_bytes", "power_usage"]

KB = 1024
MB = KB * KB
GB = MB * KB
TB = GB * KB

scale_lookup = {
    "kb": KB,
    "mb": MB,
    "gb": GB,
    "tb": TB
}

SCALES = [100, 100, 1, 1, 1, 1, 1, GB, GB, GB, GB, 1]
SLEEP_INTERVAL = 2
PROFILING = False


def convert_int(val_str):
    try:
        return float(val_str)
    except ValueError:
        postfix = val_str[-2:].lower()
        prefix = val_str[:-2]

        if postfix not in scale_lookup.keys():
            logging.warning(f"Unknown scale {postfix}")
            raise ValueError()

        value = float(prefix)

        return value * scale_lookup[postfix]


def set_scales(scale_str):
    for scale in scale_str.split(","):
        if scale.strip() == "":
            continue

        try:
            k, v = scale.split("=", 1)
        except ValueError:
            logging.warning(f"Invalid pair, must be 'k=v'.")
            continue

        try:
            col_id = COLUMNS.index(k)
        except ValueError:
            logging.warning(f"{k} is an invalid metric. Skipping...")
            continue

        try:
            SCALES[col_id] = convert_int(v)
        except ValueError:
            logging.warning(f"{v} is an invalid value. Skipping...")


def profile(func):
    def wrap(*args, **kwargs):
        if PROFILING:
            started_at = datetime.datetime.now()
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__}: {datetime.datetime.now() - started_at}")
            return result
        else:
            return func(*args, **kwargs)

    return wrap


class Plotter():

    def __init__(self, gpus, metrics, theme, refresh_rate, autoscale, **kwargs):
        self.gpu_nodes = gpus
        self.metrics: list[str] = metrics.split(",")
        self.theme = theme
        self.refresh_rate = int(refresh_rate)
        assert len(self.gpu_nodes.split(",")) <= 4

        SCALES[COLUMNS.index("power_usage")] = self.get_power_limit()

        self.reader = DCGMReader(mode="last")
        self.reader.start()

        self.auto_scale_metrics = [x for x in autoscale.split(",") if x in COLUMNS]

        self.per_entity_data = defaultdict(list[GPUStats])
        self.metric_selected_id = 0

        self._running = True
        self.key_thread = None
        self.data_thread = None

    @profile
    def get_power_limit(self):
        response = subprocess.run("nvidia-smi -q -d POWER".split(" "), stdout=subprocess.PIPE).stdout.decode("utf-8")
        possible_watts = filter(lambda x: "Current Power Limit" in x, response.split("\n"))
        for watts in possible_watts:
            watt = watts.split(":")[1]
            number = watt.strip().split(" ")[0]
            try:
                return float(number)
            except:
                pass
        raise RuntimeError("`nvidia-smi` did not provide valid `Current Power Limit` values.")

    @staticmethod
    def tryfloat(x):
        try:
            return float(x)
        except ValueError:
            return 0.0

    @profile
    def get_metric(self, gpu_name, metric_name):
        metric_id = COLUMNS.index(metric_name)
        scale = SCALES[metric_id]
        metrics: list[GPUStats] = self.per_entity_data[gpu_name]
        current_metrics = [getattr(x, metric_name) / scale for x in metrics]
        scaled_metrics = [0.0] * (30 - len(current_metrics)) + current_metrics
        return [min(x, 1) for x in scaled_metrics]

    @profile
    def print_menu(self, stdscr, width):
        start_color = '\033[43;30m'
        end_color = '\033[0m'

        metric_list_str = ""
        for i, metric_longname in enumerate(COLUMNS):
            metric = self.reader.metric_shortname_lookup[metric_longname]

            metric_list_str += " "
            if self.metric_selected_id == i:
                metric_list_str += start_color
            is_active = "O" if metric_longname in self.metrics else "X"
            metric_list_str += f"[{metric}|{is_active}]"
            if self.metric_selected_id == i:
                metric_list_str += end_color

        menu_string = f"Metrics{metric_list_str}"
        print(menu_string.rjust(width))

    @profile
    def handle_inputs(self, stdscr):
        while self._running:
            key = stdscr.getch()
            if key == ord("q"):
                self._running = False
                break
            if key == 260:  # Left arrow
                self.metric_selected_id = max(0, self.metric_selected_id - 1)
            if key == 261:  # right arrow
                self.metric_selected_id = min(len(COLUMNS) - 1, self.metric_selected_id + 1)
            if key == 10:
                metric = COLUMNS[self.metric_selected_id]
                if metric in self.metrics:
                    self.metrics.remove(metric)
                else:
                    self.metrics.append(metric)

    def handle_data(self):
        while self._running:
            gpus = self.reader.get_gpus()
            for gpu_name in gpus:
                stats = copy.copy(self.reader.get_gpu_stats(gpu_name))
                for metric in self.auto_scale_metrics:
                    value = getattr(stats, metric)
                    SCALES[COLUMNS.index(metric)] = max(value, SCALES[COLUMNS.index(metric)])

                self.per_entity_data[gpu_name].append(stats)
                if len(self.per_entity_data[gpu_name]) > 30:
                    self.per_entity_data[gpu_name] = self.per_entity_data[gpu_name][-30:]

            time.sleep(1)

    def run(self, stdscr):
        curses.curs_set(0)  # Hide the cursor
        stdscr.nodelay(True)  # Make getch non-blocking
        stdscr.clear()  # Clear the screen
        stdscr.encoding = 'utf-8'

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_YELLOW)

        self.key_thread = threading.Thread(target=partial(self.handle_inputs, stdscr))
        self.key_thread.start()
        self.data_thread = threading.Thread(target=self.handle_data)
        self.data_thread.start()

        while self._running:
            gpus = self.reader.get_gpus()
            plot_lines = self.get_plot_lines(gpus, stdscr)
            plot_lines.seek(0)

            stdscr.clearok(1)
            stdscr.refresh()
            maxy, maxx = stdscr.getmaxyx()

            lines = plot_lines.readlines()
            for y, line in enumerate(lines):
                print(line[:-1], end="")

            self.print_menu(stdscr, maxx)

            curses.napms(1000 // 30)

    @profile
    def get_plot_lines(self, gpus, stdscr):
        width = 2 if len(gpus) > 1 else 1
        height = 2 if len(gpus) > 3 else 1
        term_height, term_width = stdscr.getmaxyx()
        plot_chars = io.StringIO()
        start_time = datetime.datetime.now()
        with redirect_stdout(plot_chars):
            plt.clf()
            plt.plot_size(term_width, term_height - 1)
            plt.subplots(width, height)
            plt.theme(self.theme)
            plt.title(" ".join(self.metrics))
            for i, gpu_name in enumerate(gpus):
                x = i % 2 + 1
                y = i // 2 + 1
                for metric in self.metrics:
                    ys = self.get_metric(gpu_name, metric)
                    plt.subplot(x, y).ylim(0, 1)
                    plt.subplot(x, y).plot(ys, label=gpu_name + " " + metric)
            plt.show()
        return plot_chars

    def kill(self):
        self._running = False
        if self.key_thread is not None:
            self.key_thread.join()
        self.reader.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated gpu id list")
    parser.add_argument(
        "--metrics",
        default="power_usage",
        help=f"Comma-separated metrics list. Possible values: {COLUMNS}"
    )
    parser.add_argument("--refresh_rate", default=1000, help=f"Refresh rate.")
    parser.add_argument(
        "--theme",
        default="grandpa",
        help=f"Available plot themes https://github.com/piccolomo/plotext/blob/master/readme/aspect.md#themes"
    )
    parser.add_argument(
        "--scales",
        default="",
        help="For each metric, set the scale as a comma-separated list of a=b pairs. The scale has to be a number and can have a postfix (KB - TB)."
    )
    parser.add_argument(
        "--autoscale",
        default="",
        help="For each metric in the given comma-separated list, automatically scale height to the max encountered value."
    )
    return parser.parse_args()


def main():
    # logging.basicConfig(filename="out.txt", level=logging.INFO, format=f"%(asctime)s %(message)s")
    args = parse_args()

    set_scales(args.scales)

    plotter = Plotter(**vars(args))

    try:
        screen = curses.wrapper(plotter.run)
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
    plotter.kill()


if __name__ == "__main__":
    main()
