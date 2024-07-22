import argparse
import curses
import io
import locale
import subprocess
import threading
from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial

import plotext as plt

locale.setlocale(locale.LC_ALL, '')
code = locale.getpreferredencoding()
COLUMNS = ["GPUTL", "MCUTL", "GRACT", "SMACT", "SMOCC", "TENSO", "DRAMA", "PCITX", "PCIRX", "NVLTX",
           "NVLRX", "POWER"]
SCALES = [100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SLEEP_INTERVAL = 2


class Plotter():
    def __init__(self, gpus, metrics, theme,refresh_rate, **kwargs):
        self.popen = None
        self.gpu_nodes = gpus
        self.metrics: list[str] = metrics.split(",")
        self.theme = theme
        self.refresh_rate = int(refresh_rate)
        assert len(self.gpu_nodes.split(",")) <= 4

        SCALES[COLUMNS.index("POWER")] = self.get_power_limit()

        self.cmd = f"dcgmi dmon -e 203,204,1001,1002,1003,1004,1005,1009,1010,1011,1012,155 -i {self.gpu_nodes} -d {self.refresh_rate}".split()
        self.per_entity_data = defaultdict(list)
        self.metric_selected_id = 0

        self._running = True
        self.key_thread = None
        self.data_thread = None

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
        current_metrics = [self.tryfloat(x[metric_id]) / scale for x in self.per_entity_data[gpu_name]]
        return [0.0] * (30 - len(current_metrics)) + current_metrics

    def print_menu(self, stdscr):
        start_color = '\033[43;30m'
        end_color = '\033[0m'

        metric_list_str = ""
        for i, metric in enumerate(COLUMNS):
            metric_list_str += " "
            if self.metric_selected_id == i:
                metric_list_str += start_color
            is_active = "O" if metric in self.metrics else "X"
            metric_list_str += f"[{metric}|{is_active}]"
            if self.metric_selected_id == i:
                metric_list_str += end_color

        print(f"Metrics{metric_list_str}")

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

    def collect_data(self):
        for data_point in self.execute():
            if not self._running:
                break

            if data_point.startswith("#Entity") or data_point.startswith("ID"):
                continue
            elems = [x.strip() for x in data_point.split("  ") if x.strip() != ""]
            self.per_entity_data[elems[0]].append(elems[1:])

            for gpu_name in self.per_entity_data.keys():
                if len(self.per_entity_data[gpu_name]) > 30:
                    self.per_entity_data[gpu_name] = self.per_entity_data[gpu_name][-30:]

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

        self.data_thread = threading.Thread(target=self.collect_data)
        self.data_thread.start()

        while self._running:
            gpus = self.gpu_nodes.split(",")

            plot_lines = self.get_plot_lines(gpus, stdscr)
            plot_lines.seek(0)

            stdscr.clear()
            stdscr.refresh()
            maxy, maxx = stdscr.getmaxyx()

            lines = plot_lines.readlines()
            for y, line in enumerate(lines):
                # if y >= maxy:
                #     break
                # for x, ch in enumerate(line[:-1]):
                #     if x >= maxx:
                #         break
                #     print(ch, end="")
                #     stdscr.addstr(y, x, ch.encode(code))
                print(line[:-1], end="")

            self.print_menu(stdscr)

            stdscr.refresh()
            curses.napms(1000)

    def get_plot_lines(self, gpus, stdscr):
        width = 2 if len(gpus) > 1 else 1
        height = 2 if len(gpus) > 3 else 1
        term_height, term_width = stdscr.getmaxyx()
        plot_chars = io.StringIO()
        with redirect_stdout(plot_chars):
            plt.clf()
            plt.plot_size(term_width, term_height - 1)
            plt.subplots(width, height)
            plt.theme(self.theme)
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

    def kill(self):
        self._running = False
        if self.key_thread is not None:
            self.key_thread.join()
        if self.data_thread is not None:
            self.data_thread.join()
        self.popen.kill()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated gpu id list")
    parser.add_argument("--metrics", default="POWER", help=f"Comma-separated metrics list. Possible values: {COLUMNS}")
    parser.add_argument("--refresh_rate", default=1000, help=f"Refresh rate.")
    parser.add_argument("--theme",default="grandpa", help=f"Available plot themes https://github.com/piccolomo/plotext/blob/master/readme/aspect.md#themes")
    return parser.parse_args()


def main():
    args = parse_args()
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
