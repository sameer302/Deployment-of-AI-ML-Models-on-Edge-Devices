## Setting system baseline state for raspberry pi 5

- So in order to log performance metrics while comparing different inference scenarios, we should make sure that all the inference pipelines start from the same baseline state so that we have a fair comparison. To ensure this we need to control following parameters with respect to deployment hardware and software settings:-
    1) CPU Frequency / Governor state
        - By default the CPU scaling governor is set to work in `ondemand` mode which dynamically adjusts the CPU frequency based on system load. Idle period triggers lower frequency to save power while increase in load triggers higher frequency. 
        - Instead in order to get the best performance we should set the CPU governor to `Performance` mode which will lock the CPU at maximum frequency for consistent performance. 
        - To set the CPU governor to performance mode we need to do the following steps:
        `sudo apt install cpufrequtils`
        `sudo cpufreq-set -g performance`
        - After changing, we can check the state of CPU governor using the following command:
        `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
	- Another option is to run the below commands
	`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors`
	Then the below command,	
	`for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    		echo performance | sudo tee $cpu/cpufreq/scaling_governor
	 done`
	- After running above command we can verify using, 
	`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`

    2) Memory / Cache state
        - Linux filesystem cache and RAM reuse affect timing.
        - It is hard to control fully but we can reduce the variation by closing other running processes (browsers, vscode, etc.)
        - we should also take a snapshot of memory before starting the experiment to have a better comparison later, for this we can store the output of the command, `free -h` 
        - There is no need to clear cache memory as during the warmup period it gets settled down to the standard environment. 

    3) Warm-Up State
        - First few inferences will be slower due to cache misses, memory allocation, camera setup, etc. Hence we should have a buffer time for the inference pipeline to stabilize and then we can start with logging the performance metrics.
        - For this we can add a logic that the inference would start but the performance metrics will be logged after a duration of some fixed number of frames, for e.g., `WARMUP_FRAMES = 30`.

    4) CPU temperature
        - Raspberry Pi 5 will throttle when hot hence we need to ensure consistent starting temperature and environmental temperature before measuring the system performance. 
        - To control the hardware temperature we should make sure that starting temperature is established at a fixed range of 45 - 50 degree celsius and we also use some heatsink for our raspberry pi 5 (like active cooler fan, metal case, etc).
        - To control the starting temperature, we should follow point number `2)` and wait for some time, for e.g., 5 minutes, before starting the inference. This will ensure that our pi 5 system settles at a normal temperature and further as per point `3)` when we run some warmup inference so the pi 5 temperature eventually rises to a standard temperature value across all runs. 

    5) Background System Load
        - we should monitor if any other heavy processes are going on in the background which might steal CPU cycles, utilize memory bandwidth, IO bandwidth.
        - This can be done using commands like `htop` or `top`. 
        - We should try to make sure that before running the inference the CPU is idle and no heavy background jobs are operating.

    6) Camera State
        - Camera auto-adjustments can affect preprocessing time.
        - adjustments such as autofocus, auto exposure, auto white balance.
        - so we should have same camera settings across all runs that we are comparing.

    7) Power supply stability
        - Undervoltage affects Pi performance. This can be sue to weak adapters or interrupted power supply.
        - we can check throttling due to undervoltage using the command `vcgencmd get_throttled`
        - We should use official power adapter and charger.

    8) Resolution should be fixed. 
        - Inference cost depends on capture resolution, model input size, display resolution.
        - So we should keep them constant across runs. 
        - Also note down wat specific resolution we are using. 

    9) Benchmark duration
        - we should run inference for a significant amount of time so that firstly our system reaches a stable state and secondly we are able to see effects due to thermal throttling if any. 

    10) Display overhead
        - OpenCV displays costs CPU utilization.
        - So for fair comparison we should always have display on or off. 

## Metrics and how they are measured

### Inference performance metrics

- We should benchmark for a fixed duration of time which is quite longer so that we are able to see the efects of thermal throttling. 

- For all the metrics below we can add summary statistics which will include mean value, median value, std deviation, min/max and P95/P99 values or we can also make a box plot which shows all these at once.

1) Latency
    - end-to-end per frame processing latency starting from frame capture, preprocess, inference, postprocess, rendering, display. 
    - we calculate this per frame to analyze worst case scenario as to how long we may have to wait in the pipeline to proceed further.

2) inference latency
    - here we measure only the accelerator execution without pipeline noise.

3) FPS
    - fps is not directly the inverse of latency because there may be overlapping between capture, preprocessing, inference, rendering, due to which multiple frames can be in flight simultaneously. 
    - Hence we should treat it as a separate metric. 
    - Here we should measure end-to-end FPS for a rolling window of about 30 frames because tiny frame-time fluctuations create noisy FPS swings. 
    - why average of 30 frames ? This is because at default mode our camera sensor captures frames at 30 FPS so if we take average over 30 frames it would mean we are logging average latency every 1 second. 

4) inference FPS
    - this isolates accelerator / model capability

To add more further....

5) frame-time jitter
6) dropped frames

   
### System performance metrics

