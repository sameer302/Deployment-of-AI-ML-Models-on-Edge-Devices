import argparse
import csv
from dataclasses import fields
from email import parser
from html import parser
import time
import os
from attrs import fields
import psutil
import subprocess
from datetime import datetime

LOG_INTERVAL = 1  # seconds


def get_cpu():
    return psutil.cpu_percent(interval=None)

# psutil.cpu_percent(interval=None) returns the overall CPU utilization percentage across all cores, not just one core.
# For a 4-core system, it is an average across cores.
# If only one core is fully busy and the other three are idle, it will report around 25.0.
# If you want per-core values, use psutil.cpu_percent(interval=None, percpu=True).

def get_cpu_freq():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], text=True)
        return int(out.strip().split("=")[1]) / 1e6  # MHz
    except:
        return None
    
# vcgencmd measure_clock arm reports the shared ARM cluster clock.
# The Pi 5 CPU cores run at the same cluster frequency, not separate per-core frequencies.
    
def get_voltage():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_volts", "core"], text=True)
        return float(out.strip().split("=")[1].replace("V", ""))
    except:
        return None
    
# vcgencmd measure_volts core returns the SoC/core rail voltage for the cluster.
# The Pi does not expose separate voltage values per CPU core to standard tools.

def get_throttled_flags():
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        return out.strip().split("=")[1]  # hex flags
    except:
        return None
    
# On Raspberry Pi, throttling is reported for the SoC/ARM cluster as a whole.
# The vcgencmd get_throttled flags are not per-core; they reflect overall CPU/SoC throttling state.

def get_memory():
    return psutil.virtual_memory().percent

def get_temp():
    temps = psutil.sensors_temperatures()
    if not temps:
        return None
    for sensor in temps.values():
        if sensor:
            return sensor[0].current
    return None

def get_npu():
    try:
        p = subprocess.Popen(
            ["hailortcli", "monitor"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        time.sleep(1.2)   # allow one refresh cycle
        p.terminate()

        out, _ = p.communicate(timeout=1)

        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    util = float(parts[1])
                    fps = float(parts[2])
                    pid = int(parts[3])
                    return util, fps
                except ValueError:
                    continue

        return None, None

    except Exception:
        return None, None


# --- Hailo device handle (initialized once if needed) ---
_hailo_device = None

def _get_hailo_device():
    global _hailo_device
    if _hailo_device is None:
        try:
            from hailo_platform import Device
            _hailo_device = Device()
        except Exception as e:
            print(f"[WARNING] Could not initialize Hailo device: {e}")
            _hailo_device = False  # Mark as failed so we don't retry
    return _hailo_device if _hailo_device else None


def get_hailo_temp():
    """Get Hailo chip temperature in Celsius via ts0_temperature sensor."""
    device = _get_hailo_device()
    if device is None:
        return None
    try:
        return device.control.get_chip_temperature().ts0_temperature
    except Exception:
        return None


def get_hailo_clock():
    """Get Hailo neural network core clock rate in MHz."""
    device = _get_hailo_device()
    if device is None:
        return None
    try:
        info = device.control.get_extended_device_information()
        # Parse "Neural Network Core Clock Rate: 400.0MHz" from string representation
        for line in str(info).splitlines():
            if "Neural Network Core Clock Rate" in line:
                # Extract numeric value before "MHz"
                clock_str = line.split(":")[1].strip().replace("MHz", "")
                return float(clock_str)
        return None
    except Exception:
        return None
    
def get_running_threads_linux(pid):
    import os
    try:
        count = 0
        for tid in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                state = f.read().split()[2]
                if state == "R":
                    count += 1
        return count
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", action="store_true", help="Log CPU/system temperature")
    parser.add_argument("--cpu", action="store_true", help="Log CPU utilization")
    parser.add_argument("--memory", action="store_true", help="Log memory utilization")
    parser.add_argument("--npu", action="store_true", help="Log NPU utilization via hailortcli")
    parser.add_argument("--freq", action="store_true", help="Log CPU frequency")
    parser.add_argument("--voltage", action="store_true", help="Log CPU core voltage")
    parser.add_argument("--throttle", action="store_true", help="Log throttled status flags")
    parser.add_argument("--hailo-temp", action="store_true", help="Log Hailo chip temperature (C)")
    parser.add_argument("--hailo-clock", action="store_true", help="Log Hailo neural core clock rate (MHz)")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0 = infinite)")
    parser.add_argument("--out", type=str, default="hardware_metrics.csv", help="Path to output CSV file")
    parser.add_argument("--threads", action="store_true", help="Log total number of threads")
    parser.add_argument("--running-threads", action="store_true", help="Log running (R state) threads")
    parser.add_argument("--pid", type=int, default=None, help="PID of target process")

    args = parser.parse_args()

    csv_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    # Build CSV fields based on selected flags
    fields = ["timestamp"]
    if args.temp:
        fields.append("temperature_C")
    if args.cpu:
        fields.append("cpu_percent")
    if args.memory:
        fields.append("memory_percent")
    if args.freq:
        fields.append("cpu_freq_MHz")
    if args.voltage:
        fields.append("cpu_voltage_V")
    if args.throttle:
        fields.append("throttled_flags_hex")
    if args.npu:
        fields.append("npu_utilization_percent")
    if args.hailo_temp:
        fields.append("hailo_temp_C")
    if args.hailo_clock:
        fields.append("hailo_clock_MHz")
    if args.threads:
        fields.append("num_threads")
    if args.running_threads:
        fields.append("running_threads")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        start = time.time()
        while True:
            row = {"timestamp": datetime.now().isoformat()}

            if args.temp:
                row["temperature_C"] = get_temp()
            if args.cpu:
                row["cpu_percent"] = get_cpu()
            if args.memory:
                row["memory_percent"] = get_memory()
            if args.npu:
                util, fps = get_npu()
                row["npu_utilization_percent"] = util
            if args.freq:
                row["cpu_freq_MHz"] = get_cpu_freq()
            if args.voltage:
                row["cpu_voltage_V"] = get_voltage()
            if args.throttle:
                row["throttled_flags_hex"] = get_throttled_flags()
            if args.hailo_temp:
                row["hailo_temp_C"] = get_hailo_temp()
            if args.hailo_clock:
                row["hailo_clock_MHz"] = get_hailo_clock()
            if args.threads and args.pid:
                row["num_threads"] = psutil.Process(args.pid).num_threads()
            if args.running_threads and args.pid:
                row["running_threads"] = get_running_threads_linux(args.pid)

            writer.writerow(row)
            f.flush()
            time.sleep(LOG_INTERVAL)

            if args.duration and (time.time() - start) >= args.duration:
                break


if __name__ == "__main__":
    main()