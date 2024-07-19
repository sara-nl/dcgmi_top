import argparse
import io
import os
import time
from collections import defaultdict
from contextlib import redirect_stdout

import plotext as plt
import subprocess
import curses
import sys

COLUMNS = ["GPUTL", "MCUTL", "GRACT", "SMACT", "SMOCC", "TENSO", "DRAMA", "PCITX", "PCIRX", "NVLTX",
           "NVLRX", "POWER"]
SCALES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SLEEP_INTERVAL = 2


class Plotter():
    def __init__(self, gpus, metrics, refresh_rate, **kwargs):
        self.popen = None
        self.gpu_nodes = gpus
        self.metrics = metrics.split(",")

        self.refresh_rate = int(refresh_rate)
        assert len(self.gpu_nodes.split(",")) <= 4

        SCALES[COLUMNS.index("POWER")] = self.get_power_limit()

        self.cmd = f"dcgmi dmon -e 203,204,1001,1002,1003,1004,1005,1009,1010,1011,1012,155 -i {self.gpu_nodes} -d {self.refresh_rate}".split()
        self.per_entity_data = defaultdict(list)

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

    def execute(self):
        self.popen = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(self.popen.stdout.readline, ""):
            yield stdout_line
        self.popen.stdout.close()
        return_code = self.popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self.cmd)

    @staticmethod
    def tryfloat(x):
        try:
            return float(x)
        except ValueError:
            return 0.0

    def get_metric(self, gpu_name, metric_name):
        metric_id = COLUMNS.index(metric_name)
        scale = SCALES[metric_id]
        metrics = self.per_entity_data[gpu_name]
        if len(self.per_entity_data[gpu_name]) > 30:
            self.per_entity_data[gpu_name] = metrics[-30:]
        current_metrics = [self.tryfloat(x[metric_id]) / scale for x in self.per_entity_data[gpu_name]]
        return [0.0] * (30 - len(current_metrics)) + current_metrics

    def run(self, stdscr):
        curses.curs_set(0)  # Hide the cursor
        stdscr.clear()  # Clear the screen
        stdscr.encoding = 'utf-8'

        xs = list(range(30))

        if stdscr is not None:
            curses.start_color()
            curses.use_default_colors()

        iter_id = 0
        for data_point in self.execute():
            if data_point.startswith("#Entity") or data_point.startswith("ID"):
                continue
            iter_id += 1
            elems = [x.strip() for x in data_point.split("  ") if x.strip() != ""]
            self.per_entity_data[elems[0]].append(elems[1:])
            gpus = self.gpu_nodes.split(",")

            if iter_id % len(gpus) != 0:
                continue

            plot_lines = self.get_plot_lines(gpus, stdscr)
            plot_lines.seek(0)

            stdscr.clear()
            stdscr.refresh()

            lines = plot_lines.readlines()
            for y, line in enumerate(lines):
                print(line[:-1], end="")
                # stdscr.addstr(y, 0, line[:-1])
                # # Ensure we do not write outside the window height
                # if y >= max_y:
                #     break
                # # Truncate the line to fit within the window width
                # line = line[:max_x]
                # try:
                #     print(line[:-1], end="")
                # except curses.error:
                #     pass  # Handle any curses errors gracefully
            print(len(lines), stdscr.getmaxyx())
            # stdscr.refresh()
            # stdscr.getch()

            if stdscr is not None:
                stdscr.refresh()
                stdscr.timeout(self.refresh_rate)
            else:
                time.sleep(self.refresh_rate / 1000)

    def get_plot_lines(self, gpus, stdscr):
        width = 2 if len(gpus) > 1 else 1
        height = 2 if len(gpus) > 3 else 1
        term_height, term_width = stdscr.getmaxyx()
        plot_chars = io.StringIO()
        with redirect_stdout(plot_chars):
            plt.clf()
            plt.plot_size(term_width, term_height - 1)
            plt.subplots(width, height)
            plt.theme("grandpa")
            plt.title(" ".join(self.metrics))
            for i, gpu_id in enumerate(gpus):
                x = i % 2 + 1
                y = i // 2 + 1
                gpu_tag = f"GPU {gpu_id}"
                for metric in self.metrics:
                    ys = self.get_metric(gpu_tag, metric)
                    plt.subplot(x, y).ylim(0, 1)
                    plt.subplot(x, y).plot(ys, label=gpu_tag + " " + metric)
            plt.show()
        return plot_chars


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated gpu id list")
    parser.add_argument("--metrics", default="POWER", help=f"Comma-separated metrics list. Possible values: {COLUMNS}")
    parser.add_argument("--refresh_rate", default=1000, help=f"Refresh rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    plotter = Plotter(**vars(args))

    try:
        # plotter.run(None)
        # TODO: Fix this
        screen = curses.wrapper(plotter.run)
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
    plotter.popen.kill()


if __name__ == "__main__":
    main()
