# Jetlink for iPhone

Runs the large (chestnut) driving models for a comma on an iPhone's Neural
Engine, and serves them to the comma over a wired link. It speaks the same
protocol as the Jetson and Mac servers, over TCP, so the comma's side is
zoompilot's existing `JetlinkEndpoint` mode.

**Status: experimental. Not yet driven.** On an iPhone 17 Pro (A19 Pro) the
766 MB model takes 17.7 ms a frame on the Neural Engine at 20 frames a
second, 21.5 ms p99, with no frame over 35 ms in a minute. The link to the comma is not working yet: see
[Connecting to the comma](#connecting-to-the-comma).

## What you need

- An iPhone with USB-C and 8 GB of memory: an iPhone 15 Pro or a later Pro.
  These have USB 3 and the strongest Neural Engines.
- iOS 17 or later, and a Mac with Xcode 26 to build and install the app.
- A comma running zoompilot's `jetson-trt` branch, with the patch in
  `comma/zoompilot-tcp-provisioning.patch` (see below).
- A wired link. Wi-Fi misses the 50 ms frame budget.
- Power for the phone. Its only port carries the link: charge it with
  MagSafe, or through a hub with power pass-through.

## Install with a free Apple account

A free Apple ID is enough to run the app on your own phone.

1. On the iPhone: Settings > Privacy & Security > Developer Mode, turn it on,
   and let the phone restart. The switch appears once the phone has been
   plugged into a Mac with Xcode open.
2. In Xcode > Settings > Accounts, add your Apple ID. It shows as a
   "Personal Team". To find its ten-character team ID, choose that team once
   under the Jetlink target's Signing & Capabilities, then
   `grep -m1 DEVELOPMENT_TEAM ios/Jetlink.xcodeproj/project.pbxproj`, and
   `git checkout ios/Jetlink.xcodeproj` to undo the change.
3. Create `ios/Config/Local.xcconfig` with your team and a bundle identifier
   of your own. Git ignores this file, so your team never lands in a commit:

   ```
   DEVELOPMENT_TEAM = ABCDE12345
   PRODUCT_BUNDLE_IDENTIFIER = com.yourname.jetlink
   ```

   Set them here rather than in Xcode's Signing & Capabilities, which would
   write them into the project file.
4. `open ios/Jetlink.xcodeproj`, plug in the iPhone, pick it as the run
   destination, and press Run. Run builds Release, as the car needs. The
   first build downloads onnxruntime 1.29.0 (64 MB); Xcode may ask to install
   the iOS platform (Settings > Components).
5. Trust yourself as the developer on the phone: Settings > General > VPN &
   Device Management > your Apple ID > Trust.

With a free account the app stops opening **7 days** after you install it.
Press Run again to renew it; models and settings are kept. Check the date
before a drive.

The app asks for the Increased Memory Limit capability, which a free team
can sign: a 766 MB model and its CoreML compile need more memory than iOS
gives an app by default. The Status tab shows how much memory the app may
still use.

## Using it

1. Open Jetlink. The server starts on port 5599, and the Status tab lists
   the phone's addresses. Keep the app on screen: iOS suspends background
   apps, and a suspended app serves nothing. The app keeps the screen on
   while serving.
2. On the Models tab, download the model the comma drives with (the one
   picked under Big Model on the comma), on Wi-Fi, and tap Prepare. The comma
   can upload it instead, but that takes minutes in the car.
3. Status > Benchmark > **Run for 1 minute**: the model at the comma's 20
   frames a second. You want p99 at 35 ms or less and nothing over 50 ms; the
   rest of the 50 ms is for the cable. **Run for 10 minutes**, set up as in
   the car (charging, in its mount), shows how much the phone slows as it
   warms.
4. **Accuracy.** The Benchmark screen gives a command to run on a Mac on the
   same Wi-Fi: `scripts/verify_parity.py` against the phone, compared with
   onnxruntime on the Mac. It must end with OK.

## Connecting to the comma

The comma's usual Jetlink link is a vendor USB device read with libusb.
iOS gives apps no access to one, and there is no USB driver kit on iPhone.
iOS does drive USB network adapters itself, so the phone is a TCP server and
the comma connects to it. zoompilot already has that mode: set the
`JetlinkEndpoint` param to `host:port`, and its client connects over TCP and
its gadget owner leaves the USB port alone.

### The fork patch

In TCP mode zoompilot's gadget owner never starts the parked provisioning
run, because it waits for a gadget that TCP mode never presents. That run
records the engine as ready, drives the home icon, and compiles the camera
warp; without the warp, modeld never takes the large model. Every borrow of
the endpoints also waited out an 8 s timeout first.
`comma/zoompilot-tcp-provisioning.patch` fixes both, in `owner.py` and
`lending.py`, with tests. The fork's owner and lending tests pass with it
(50), and its new tests fail without it.

Apply it to the fork, or for a quick test on the comma itself, with updates
off so the updater does not reset it:

```bash
cd /data/openpilot && git apply /data/zoompilot-tcp-provisioning.patch
echo -n 1 > /data/params/d/DisableUpdates
```

### Order of the steps

The owner does not let go of a gadget it already holds when the endpoint
changes, so:

1. `echo -n 192.168.60.2:5599 > /data/params/d/JetlinkEndpoint`
2. Settings > Models > Accelerator Link **off**, which releases the port.
3. Bring the link up (below).
4. On the iPhone: Settings > Ethernet > (the adapter) > Configure IP >
   Manual, `192.168.60.2`, subnet mask `255.255.255.0`, no router. Check
   from the comma with `ping -c 3 192.168.60.2`.
5. Accelerator Link **on**. The app shows Comma: Connected.

### The link

**Ethernet (recommended).** A USB-C Ethernet adapter on the comma (Realtek
RTL8152/8153 or ASIX AX88179; AGNOS has those drivers), a USB-C hub with
Ethernet and power pass-through on the phone, and a cable between them. Give
the comma's adapter `192.168.60.1/24`. No power flows between the two.

**One USB cable, through a hub.** The comma presents itself as a USB
network adapter (CDC-NCM) with `comma/setup_net_gadget.sh`. On a comma with
AGNOS kernel 4.9.103, `--check` found NCM built in
(`CONFIG_USB_CONFIGFS_NCM=y`). But plugging an iPhone into the comma's USB-C
port rebooted the comma: its charger logged "Weak charger detected" and
"Reverse boost detected", and the kernel log stopped at a Type-C change
with no crash message, which is a power loss rather than a software fault.
Over a C-to-C cable the two negotiate who powers whom, and the comma ended
up supplying the phone. iPhone > USB-C hub > USB-A to USB-C cable > comma
keeps the comma up: a USB-A port only supplies power, so the comma never
sources it. Whether iOS then brings the adapter up, and the link's speed,
are not yet measured.

The script releases jetlink's own FunctionFS gadget, which zoompilot sets up
at boot and which holds the port. Its teardown leaves "error: gadget torn
down" in `/dev/shm/jetlink-gadget`, and zoompilot reads any error there as
no link at all, TCP endpoint included, so the script clears it once its own
gadget is up.

```bash
sudo bash setup_net_gadget.sh --check      # changes nothing; says whether NCM exists
sudo bash setup_net_gadget.sh              # the adapter, as 192.168.60.1
sudo bash setup_net_gadget.sh --teardown
```

The gadget does not survive a reboot. Bringing it up at boot, in place of
the FunctionFS gadget whenever `JetlinkEndpoint` is set, is the fork change
to make once the cable works.

### Speed over the link

Parked, with the link up:

- The Benchmark screen's "Over the link, from the comma" command runs
  `scripts/bench_link.py` on the comma against the phone. The fork already
  carries this repository as `jetlink_repo`. Turn Accelerator Link off for
  it, so the comma's own client is not holding the phone's server.
- zoompilot's `tools/jetlink_live_bench.sh 180` runs the real cameras and
  modeld, without controls, for three minutes.

## Neural Engine, GPU, or Automatic

Settings > Run models on:

- **Automatic** (default). The first time a model is prepared, Jetlink
  builds it for the Neural Engine and the GPU, runs each at 20 frames a
  second, and keeps the Neural Engine if its p99 is 35 ms or less, since it
  does the same work for far less power; otherwise the GPU if it makes
  35 ms; otherwise the faster. The other build is deleted. "Measure again"
  on the Models tab repeats it.
- **Neural Engine**: prepared so the whole model stays on the Neural Engine
  (below), with Apple's FastPrediction specialization, and with one CPU
  core kept busy while frames arrive (Settings > Keep the CPU ready between
  frames).
- **GPU**: CoreML on the GPU, with the Metal keep-alive the Mac server uses.
  On an iPhone 17 Pro it measured 154 ms p99, too slow to drive.

## How it works

`JetlinkKit` is the server, in Swift, built for iOS and macOS:

| File | What it ports |
| --- | --- |
| `WireProtocol.swift`, `Transport.swift` | `jetlink/protocol.py`, `transport/base.py`, `transport/tcp.py` |
| `Session.swift`, `EngineHost.swift`, `Server.swift` | `server/session.py`, `_serve` in `server/main.py` |
| `Queues.swift`, `Convert.swift` | `jetlink/queues.py` |
| `ModelSpec.swift`, `Onnx.swift`, `Pickle.swift` | `jetlink/spec.py`, `jetlink/onnx_meta.py` |
| `OnnxPrepare.swift`, `Proto.swift` | `jetlink/onnx_patch.py`, `_prepared_model`, `scripts/ane_passes.py` |
| `OrtBackend.swift`, `OrtEngine.swift`, `MetalKeepAlive.swift`, `CPUKeepWarm.swift` | `server/backends/ort/` |
| `EngineCache.swift`, `Registry.swift` | `server/cache.py`, `jetlink/registry/` |
| `Benchmark.swift` | the in-app benchmark |

onnxruntime is Microsoft's 1.29.0 build for iOS and macOS, the version the
Mac backend is measured with, through a small C shim (`COrtShim`). The comma
uploads raw ONNX, so the phone prepares it itself. `OnnxPrepare` maps the
file, copies untouched weights straight through, and transposes the Gemm
weights one at a time as it writes, so the model is never in memory twice.

Differences from the Python server:

- **The Neural Engine build.** The Mac's runs the policy's LayerNorms in
  fp32, which the Neural Engine cannot do, and leaves two `Expand` ops CoreML
  will not take. Apple's MLComputePlan showed the model split in two with
  the whole policy on the GPU: cheap on a Mac, slow on a phone. The phone
  rewrites the `Expand`s as the equivalent `Tile`s, so the model is one
  CoreML program, and divides each policy LayerNorm's input by 8 in fp16.
  LayerNorm is unchanged by the scale up to epsilon, and the scale keeps the
  fp16 squares in range: the inputs reach 1,189, whose square overflows
  fp16. 99.5% of the estimated cost lands on the Neural Engine.
- **The small heads in fp32.** The layers between the end of the vision
  trunk and the output (24 nodes, 4 MB of weights in the 766 MB model) are
  cast to fp32, so CoreML places them on the GPU or CPU. In fp16 on an
  iPhone 17 Pro's Neural Engine, `road_transform` failed the parity gate
  (worst column 0.9989). Computed exactly from the Neural Engine's own
  vision output, every column is 0.9996 or better. The Mac's speed did
  not change.
- **A CPU core kept busy on the Neural Engine path** while frames arrive.
  Paced at 20 Hz, the CPU's clocks drop between frames and CoreML's side of
  each prediction runs slowly; keeping the CPU busy took p99 from 54 to
  36 ms on an M1 Pro. Keeping the Neural Engine busy did not help.
- **Conversions through vImage.** Each frame converts about 800,000 camera
  bytes to float16. As element loops that is 0.2 ms optimized and 63 ms
  unoptimized; vImage is 0.07 ms either way, bit for bit the same.
- No worker process: Swift has no GIL. The session is created on the build
  thread and run on the serving thread, with inputs and outputs bound to
  fixed buffers once.
- After a build only CoreML's compiled model is kept: 1.4 GB instead of
  2.2 GB for a 766 MB model, with outputs bit for bit the same.
- A shutdown request from the comma is answered `ok: false`: the comma
  cannot power a phone off. The phone reports no sensor telemetry, as the
  Mac does not.
- Where the Python falls back to onnx's shape inference, there is none: the
  Neural Engine passes refuse a model missing a shape they need rather than
  compute garbage, and the Gemm rewrite leaves that MatMul, which only makes
  the weight slower to load. The driving models record every shape.

## Verified

On a 16 GB M1 Pro, macOS 26.5, with the 766 MB model `09d080f36965bb2a`:

- **Preparation**: the Swift and Python preparations are the same protobuf
  message, field for field, for the GPU, Neural Engine and CPU paths
  (`scripts/check_prepare.py`), and on synthetic models that exercise each
  branch (`scripts/prepare_variants.py`).
- **Queues and conversions**: bit for bit the Python's, over runs that wrap
  every ring, with resets, NaN, overflow and subnormal inputs.
- **End to end**: `scripts/bench_link.py` and `scripts/verify_parity.py`,
  unmodified, against `jetlink-swift serve`: upload, build, load, serve.

  | | GPU | Neural Engine |
  | --- | ---: | ---: |
  | worst column correlation | 0.99957 | 0.99956 |
  | mean error, `plan` / `lead_prob` | 0.0046 / 0.0150 | 0.0073 / 0.0346 |
  | round trip at 20 Hz, mean / p99 | 48.0 / 60.2 ms | 32.5 / 36.9 ms |
  | frames over 50 ms | 9.7% (Python server, same Mac) | 0 of 1,190 |

  The Neural Engine build passes the parity gate with less margin than the
  Mac's own (whose `lead_prob` error is 0.0149), from the policy's
  LayerNorms running in fp16.
- **Restart**: a restarted server preloads the last engine and serves a
  client that sends only the model's identity, as modeld does.
- `swift test` runs 23 tests, including the whole protocol over TCP against
  the Python reference.

On an iPhone 17 Pro, the in-app benchmark (1,201 frames at 20 Hz, Release,
CPU keep-warm on): 17.7 ms mean, 21.5 ms p99, 33.9 ms max a frame, of which
17.5 ms is the model on the Neural Engine and 0.2 ms the queues. No frame
over 35 ms; nominal temperature throughout. The parity gate, run from a Mac
over Wi-Fi against the phone: first failed on `road_transform` (worst column
0.9989), and passes with the fp32 heads (every column 0.99954 or better;
`lane_lines_prob` mean error 0.080 before, 0.009 after).

## Not yet known

- **The link to the comma**: over Ethernet, and whether the one-cable link
  can be made safe for the comma's power.
- **The phone's timings with the fp32 heads**: the numbers above were
  measured just before that change; re-run the benchmark.
- **Memory**: the Mac server's peak while building is 3.0 GB; the 1.7 GB
  Lebowski model may not fit an 8 GB phone.
- **In the car**: heat over a long drive, link drops, and the app being
  interrupted by a call or notification.

## Developing

```bash
make -C ios test      # JetlinkKit's tests
make -C ios check     # Swift preparation against the Python (needs the repo's .venv with onnx)
make -C ios device    # build for a phone, unsigned
make -C ios server    # ios/JetlinkKit/.build/release/jetlink-swift
make -C ios project   # regenerate Jetlink.xcodeproj from project.yml (xcodegen)

ios/JetlinkKit/.build/release/jetlink-swift serve --device ane --port 5599
python3 scripts/bench_link.py --host 127.0.0.1 --onnx big.onnx --rate 20
ios/JetlinkKit/.build/release/jetlink-swift build big.onnx --device ane --bench 60
python3 ios/scripts/check_prepare.py --onnx big.onnx --swift ios/JetlinkKit/.build/release/jetlink-swift
```

`scripts/queue_fixtures.py` and `scripts/server_fixtures.py` regenerate the
test fixtures from the Python package.
