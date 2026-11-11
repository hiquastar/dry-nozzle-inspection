# AFGNet final model on the Raspberry Pi 5

Deployment measurement of Chapter 5 (Section "Embedded Deployment") for the final
model of Chapter 6 (configuration 7, seeds 42 and 123). No PyTorch: ONNX Runtime,
numpy, OpenCV and psutil only.

| Path | Content |
|---|---|
| `models/afgnet_final_s{42,123}.onnx` | the network, float32, input `[1,3,480,640]` |
| `models/angle_reader_s{42,123}.onnx` | the learned angle reader, input the mist map + tip + axis of each spray lance |
| `config.json` | thresholds, normalisation and gate threshold per seed |
| `pipeline.py` | the reported system: preprocess, forward, decode + NMS, mist gate and angle reader |
| `benchmark.py` | the Chapter 5 protocol: 20 warm-up frames, 100 timed frames |
| `data/video_cam3_v2.mp4` | operating video, CAM3 V2 test recording, 2048x1536, 5 fps (the video of the earlier v9 benchmark) |
| `reference/video_s{42,123}.json` | outputs of this package on the development PC, for the agreement check |

The ONNX graphs match PyTorch to about 1e-5; the PC check over the whole test set is in
`train/model_2/results/onnx_parity_test_final.json`.

## Run

From a USB stick (copy to the home folder first: the stick is FAT/exFAT, where the
virtual environment and `chmod` do not work):

```bash
cp /media/$USER/<usb-name>/pi5_final.zip ~/
cd ~ && unzip pi5_final.zip
cd ~/pi5_final && chmod +x run.sh && ./run.sh
```

Or over the network:

```bash
scp -r pi5_final pi@<pi-ip>:~/
ssh pi@<pi-ip>
cd ~/pi5_final && chmod +x run.sh && ./run.sh
```

The first run creates `.venv` and installs `requirements.txt` (Pi OS Bookworm or later,
64 bit); it needs an internet connection once. `run.sh` runs seed 42, waits 60 s, then runs seed 123. One seed by hand:

```bash
. .venv/bin/activate
python3 benchmark.py --seed 42
```

Before the run: close other programs, note whether the active cooler is fitted, and
let the board idle a few minutes.

**Wall power** (inline meter): note the reading of the idle board, and the typical
reading during the timed run. The run is short; for a steadier reading run
`python3 benchmark.py --seed 42 --count 600 --out results/power_s42.json` (about a
minute) and read the meter during it. The values can be passed with
`--power-idle-w` and `--power-w`, or sent together with the JSON files.

## Output

`results/pi5_s{seed}.json`, with

- `time_ms`: decode, preprocess, forward, decode_nms, readers, total (mean, median, p95, min, max)
- `fps`: sustained over the 100 frames, and 1000 / mean total
- `cpu_percent_4_cores` (mean, peak), `cpu_percent_process`, `rss_mb` (mean, peak)
- `soc_temp_c` (idle, start, end, peak), `cpu_mhz` (min, max), `throttled` (vcgencmd)
- `agreement_with_pc`: frames where the Pi reports other lances, states or angles than the PC (should be 0)
- `platform` and `per_frame_ms`

Send back both `results/pi5_s42.json` and `results/pi5_s123.json` (and the power readings).
