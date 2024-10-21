import multiprocessing
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from typing import Optional


class GPUStats:
    gpu_utilization = 0
    mem_copy_utilization = 0
    gr_engine_active = 0
    sm_active = 0
    sm_occupancy = 0
    tensor_active = 0
    dram_active = 0
    pcie_tx_bytes = 0
    pcie_rx_bytes = 0
    nvlink_tx_bytes = 0
    nvlink_rx_bytes = 0
    power_usage = 0

    _n_samples: int = 0


def _dcgmi_splitter(line):
    return [x.strip() for x in line.split('  ') if len(x.strip()) > 0]


def sdir(obj):
    """
    Returns a list of public attributes of the given object.

    Args:
        obj: The object to get public attributes from.

    Returns:
        A list of public attribute names.
    """
    return [x for x in dir(obj) if not x.startswith('_')]


def metric_shortnames(*args):
    out = subprocess.check_output('dcgmi dmon -l'.split(' '))
    table: dict[str, str] = {}
    for line in out.decode().split('\n')[3:-1]:
        long_name, short_name, field_id = _dcgmi_splitter(line)
        table[long_name] = short_name
    return table


def metric_lookup(*args):
    out = subprocess.check_output('dcgmi dmon -l'.split(' '))
    table: dict[str, str] = {}
    for line in out.decode().split('\n')[3:-1]:
        long_name, short_name, field_id = _dcgmi_splitter(line)
        table[long_name] = field_id
    try:
        return ','.join(table[arg] for arg in args[0] if arg != 'n_samples')
    except KeyError:
        print(f"Not all metric names were found. Available metrics: {', '.join(table.keys())}")


class DCGMReader:
    def __init__(self, mode="means"):
        assert mode in ["means", "last"]
        self.mode = mode
        self.process = None
        self._gpu_stats: dict[str, GPUStats] = defaultdict(GPUStats)
        self._stats_lock = multiprocessing.Lock()
        self.thread: Optional[threading.Thread] = None

        self.metric_names = sdir(GPUStats)
        self.metric_ids = metric_lookup(self.metric_names)
        self.metric_shortname_lookup = metric_shortnames(self.metric_names)
        self.refresh_rate_ms = 250

    def reset(self):
        with self._stats_lock:
            for gpu_mean in self._gpu_stats.values():
                for key in sdir(gpu_mean):
                    setattr(gpu_mean, key, 0)

                gpu_mean._n_samples = 0

    def _collect(self):
        cmd = f'dcgmi dmon -d {self.refresh_rate_ms} -e {self.metric_ids}'
        self.process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )
        for line in iter(self.process.stdout.readline, ''):
            if line.startswith('#') or line.startswith('ID'):
                continue

            values = _dcgmi_splitter(line)
            gpu_name = values[0]
            self._gpu_stats[gpu_name]._n_samples += 1
            n_samples = self._gpu_stats[gpu_name]._n_samples
            with self._stats_lock:
                for metric_name, s_value in zip(self.metric_names, values[1:]):
                    value = float(s_value)
                    if self.mode == 'means':
                        cur_mean = getattr(self._gpu_stats[gpu_name], metric_name)
                        setattr(
                            self._gpu_stats[gpu_name],
                            metric_name,
                            cur_mean - (cur_mean / n_samples) + (value / n_samples)
                        )
                    elif self.mode == 'last':
                        setattr(self._gpu_stats[gpu_name], metric_name, value)

    def start(self):
        self.thread = threading.Thread(target=self._collect)
        self.thread.daemon = True
        self.thread.start()
        self.reset()

    def stop(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        self.thread.join()

    def get_gpus(self):
        """
        Returns a list of available GPU names.

        Returns:
            list: A list of available GPU names.
        """
        return list(self._gpu_stats.keys())

    def get_gpu_stats(self, gpu_name) -> GPUStats:
        """
        Retrieves GPU statistics for the specified GPU name.

        Args:
            gpu_name (str): The name of the GPU.

        Returns:
            dict or None: The GPU statistics if found, otherwise None.
        """
        return self._gpu_stats.get(gpu_name, None)


def main():
    reader = DCGMReader()
    reader.start()

    time.sleep(2)
    print(reader.get_gpus())
    print(reader.get_gpu_stats(reader.get_gpus()[0]).__dict__)

    reader.stop()


if __name__ == '__main__':
    main()
