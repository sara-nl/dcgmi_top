import argparse
import os
import time
from collections import defaultdict

import plotext as plt
import subprocess
import curses
import sys

COLUMNS = ["GPUTL", "MCUTL", "GRACT", "SMACT", "SMOCC", "TENSO", "DRAMA", "PCITX", "PCIRX", "NVLTX",
           "NVLRX", "POWER"]
SLEEP_INTERVAL = 2


class Plotter():
    def __init__(self, gpus, metrics, refresh_rate, **kwargs):
        self.popen = None
        self.gpu_nodes = gpus
        self.metrics = metrics.split(",")

        self.refresh_rate = int(refresh_rate)
        assert len(self.gpu_nodes.split(",")) <= 4
        self.cmd = f"dcgmi dmon -e 203,204,1001,1002,1003,1004,1005,1009,1010,1011,1012,155 -i {self.gpu_nodes} -d {self.refresh_rate}".split()

    def execute(self):
        self.popen = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(self.popen.stdout.readline, ""):
            yield stdout_line
        self.popen.stdout.close()
        return_code = self.popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self.cmd)

    def run(self, stdscr):
        per_entity_data = defaultdict(list)
        xs = list(range(30))

        def tryfloat(x):
            try:
                return float(x)
            except ValueError:
                return 0.0

        def get_metric(gpu_name, metric_name):
            metric_id = COLUMNS.index(metric_name)
            metrics = per_entity_data[gpu_name]
            if len(per_entity_data[gpu_name]) > 30:
                per_entity_data[gpu_name] = metrics[-30:]
            current_metrics = [tryfloat(x[metric_id]) for x in per_entity_data[gpu_name]]
            return [0.0] * (30 - len(current_metrics)) + current_metrics

        if stdscr is not None:
            curses.start_color()
            curses.use_default_colors()

        iter_id = 0
        for data_point in self.execute():
            if data_point.startswith("#Entity") or data_point.startswith("ID"):
                continue
            iter_id += 1
            elems = [x.strip() for x in data_point.split("  ") if x.strip() != ""]
            per_entity_data[elems[0]].append(elems[1:])
            gpus = self.gpu_nodes.split(",")

            if iter_id % len(gpus) != 0:
                continue

            width = 2 if len(gpus) > 1 else 1
            height = 2 if len(gpus) > 3 else 1

            plt.clf()
            plt.subplots(width, height)
            plt.clear_terminal(os.get_terminal_size().lines)
            plt.theme("dark")
            plt.title(" ".join(self.metrics))
            for i, gpu_id in enumerate(gpus):
                x = i % 2 + 1
                y = i // 2 + 1
                gpu_tag = f"GPU {gpu_id}"
                for metric in self.metrics:
                    ys = get_metric(gpu_tag, metric)
                    plt.subplot(x, y).plot(ys, label=gpu_tag + " " + metric)
            plt.show()

            if stdscr is not None:
                stdscr.refresh()
                curses.napms(self.refresh_rate)
            else:
                time.sleep(self.refresh_rate / 1000)


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
        plotter.run(None)
        # TODO: Fix this
        # screen = curses.wrapper(plotter.run)
    except Exception as e:
        print(e)
        plotter.popen.kill()


if __name__ == "__main__":
    main()
