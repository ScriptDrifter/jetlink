# Jetlink, with an iPhone server

This is a fork of [zoompilot/jetlink](https://github.com/zoompilot/jetlink)
that adds **Jetlink for iPhone**: an iOS app that runs openpilot's large
(chestnut) driving models on an iPhone's Neural Engine and serves them to a
comma. Everything upstream (the Mac app, Jetson, Linux) is unchanged and
documented [below](#jetlink-upstream).

> **Experimental, and not yet driven.** The app runs the model within
> budget on an iPhone 17 Pro, but the link from the phone to the comma has
> not been validated, and nothing here has been tested in a car. The comma's
> small model drives whenever the link is down.

- [What this fork adds](#what-this-fork-adds)
- [How the iPhone app works](#how-the-iphone-app-works)
- [Performance](#performance)
- [Install the app with Xcode and a free Apple account](#install-the-app-with-xcode-and-a-free-apple-account)
- [Connect it to the comma](#connect-it-to-the-comma)
- [zoompilot jetson-trt: limitations and the patch](#zoompilot-jetson-trt-limitations-and-the-patch)

## What this fork adds

Everything new is under [`ios/`](ios/):

| Path | What it is |
| --- | --- |
| `ios/Jetlink/` | The SwiftUI app: status, models, benchmark, logs, settings |
| `ios/JetlinkKit/` | The Jetlink server in Swift, for iOS and macOS: the wire protocol, TCP transport, history queues, ONNX preparation, and onnxruntime with CoreML |
| `ios/JetlinkKit/Sources/jetlink-swift/` | The same server as a Mac command, so the repository's Python tools can test it |
| `ios/comma/zoompilot-tcp-provisioning.patch` | A fix for zoompilot's TCP mode, needed on the comma (see [below](#zoompilot-jetson-trt-limitations-and-the-patch)) |
| `ios/comma/setup_net_gadget.sh` | Makes the comma's USB-C port a USB network adapter, for a one-cable link (experimental) |
| `ios/scripts/` | Checks of the Swift server against the Python one, and test fixtures |
| [`ios/README.md`](ios/README.md) | The detailed reference |

Outside `ios/`, the fork changes only this README (the iPhone sections
above upstream's), a line in [docs/status.md](docs/status.md), and
`scripts/deploy_to_comma.sh`, which now skips the iOS build output.

## How the iPhone app works

The comma sends each prepared camera frame to the phone, the phone runs the
model, and the prediction goes back, 20 times a second, exactly as with a Mac
or a Jetson. The protocol is upstream's, unchanged.

**The link is TCP.** Upstream's link is a vendor-specific USB device that
the server reads with libusb. iOS gives apps no access to such a device, and
iPhones have no USB driver kit. iOS does drive USB network adapters itself,
so the phone is a TCP server on port 5599, and the comma connects to it with
zoompilot's existing `JetlinkEndpoint` setting, over Ethernet adapters or a
USB network link.

**The server is a Swift port of upstream's Python server** (`JetlinkKit`),
because a phone cannot run the Python one. It runs onnxruntime 1.29.0 with
CoreML, the same runtime and version the Mac server is measured with. The
comma uploads the raw ONNX, so the phone prepares it itself. The Swift
preparation produces the same model, protobuf field for field, as upstream's
`onnx_patch.py`, which `ios/scripts/check_prepare.py` checks.

**It runs on the Neural Engine.** The Mac server's Neural Engine build
leaves part of the model to the GPU. Apple's CoreML placement report showed
the model split in two, with the whole policy half on the GPU. That is cheap
on a Mac and slow on a phone, whose GPU measured 154 ms p99 on its own. So
the iPhone build differs in five ways:

- **One piece, on the Neural Engine.** Two `Expand` ops CoreML would not
  take are rewritten as the equivalent `Tile`s, so the model is one CoreML
  program. The policy's LayerNorms run in fp16 on the Neural Engine with
  their inputs divided by 8 first. LayerNorm gives the same result for a
  scaled input, and the scale keeps fp16 from overflowing: those inputs
  reach 1,189, whose square is past fp16's maximum of 65,504. 99.5% of the
  model's estimated cost now runs on the Neural Engine.
- **The small heads in fp32.** Between the end of the vision network and
  the output are a few small layers (24 nodes, 4 MB of weights) that make
  `road_transform`, `pose`, `lane_lines_prob` and the like. On the Neural
  Engine they ran in fp16, and on an iPhone 17 Pro `road_transform` failed
  the accuracy gate (worst column 0.9989, gate 0.999). They now run in fp32,
  which CoreML places on the GPU or CPU. The phone now passes the gate:
  `road_transform`'s worst column is 0.99954, and the error on
  `lane_lines_prob` fell from 0.080 to 0.009.
- **The CPU kept ready.** Between frames the CPU drops its clocks, and
  CoreML's share of the next frame then runs slowly. While frames arrive,
  the app keeps one CPU core busy; it stops a second after the last frame.
  It can be turned off in Settings.
- **Apple's FastPrediction** hint for the Neural Engine.
- **Vectorized frame conversions.** Each frame converts about 800,000 camera
  bytes to fp16, now in 0.07 ms.

Settings > Run models on offers **Automatic** (the default), **Neural
Engine** or **GPU**. Automatic builds a model for both, times each at 20
frames a second, and keeps the Neural Engine whenever its p99 is 35 ms or
less, because it does the same work for much less power.

## Performance

The comma allows 50 ms per frame, cable included.

**iPhone 17 Pro (A19 Pro)**, 766 MB model, the in-app benchmark: 1,201
frames at 20 a second over 60 seconds, Release build, CPU keep-warm on:

| Current build, Neural Engine | mean | p99 | max |
| --- | ---: | ---: | ---: |
| Whole frame (the phone's share) | 17.7 ms | 21.5 ms | 33.9 ms |
| The model on the Neural Engine | 17.5 ms | 21.3 ms | 33.8 ms |
| History queues | 0.2 ms | 0.3 ms | 0.6 ms |
| Frames over 35 ms / over 50 ms | 0 | 0 | |

The phone stayed at nominal temperature, and every 10-second window had a
p99 between 17.4 and 23.2 ms. A 10-minute run shows whether it holds as
the phone warms.

**With the comma, over one cable** (iPhone > USB-C hub > USB-A to USB-C
cable > comma, the comma as a USB network adapter), parked:

| | median | p99 | max |
| --- | ---: | ---: | ---: |
| `bench_link.py`: round trip, 1,190 frames at 20 Hz | 35.2 ms | 46.5 ms | 60.7 ms |
| zoompilot's live bench: modeld's frame, 3,431 frames over 180 s | 35.1 ms | 39.1 ms | 111 ms |

In the live bench the comma's real cameras and modeld ran with controls
off. Every frame came from the large model on the phone, with none dropped
and one over 50 ms. The comma's own small model takes 24.9 ms a frame there.
About 19 ms of each round trip is the link, although it runs at USB 3.0
(the comma's controller reports `super-speed`), where the frame's 533 KB
would take about 1 ms on the wire. The rest is overhead on the way, most
likely the comma's network driver or socket buffers, and is not yet
tracked down.

The first build, with the policy on the GPU, measured 66 ms p99 on the
Neural Engine setting and 154 ms on the GPU.

**M1 Pro Mac**, the same model, through the server with upstream's own
`bench_link.py` and `verify_parity.py`:

| | Python server, GPU (upstream) | Swift server, Neural Engine |
| --- | ---: | ---: |
| Round trip at 20 Hz, mean / p99 | 47.9 / 64.3 ms | 32.5 / 36.9 ms |
| Frames over 50 ms | 38 of 390 (9.7%) | 0 of 1,190 |
| Accuracy gate (every output column correlated at 0.999 or better) | passes | passes (worst column 0.99956) |

The Neural Engine build passes the accuracy gate with less margin than the
Mac's own: the error on `lead_prob` roughly doubles, from running the
policy's LayerNorms in fp16. Moving the small heads to fp32 left the Mac's
speed where it was (in-process benchmark: 32.4 ms mean, 38.2 ms p99). On the
iPhone 17 Pro the accuracy gate passes (every column 0.99954 or better). The
timings above were measured just before that change. Check your own phone
with the accuracy command on the app's Benchmark screen.

These changes are not in the Mac app, which runs upstream's Python server.

## Install the app with Xcode and a free Apple account

You need a Mac with **Xcode 26**, an iPhone with **USB-C and 8 GB of
memory** (an iPhone 15 Pro or a later Pro) on **iOS 17 or later**, and an
**Apple ID**. A free account is enough; no paid developer membership.

### 1. Get the code

```bash
git clone -b ios-app https://github.com/ScriptDrifter/jetlink.git
cd jetlink
```

### 2. Prepare Xcode and the iPhone

1. Open Xcode once. If it offers to install the **iOS platform**, accept.
   It is also under Xcode > Settings > Components.
2. **Xcode > Settings > Accounts**: click **+**, choose **Apple ID**, and
   sign in. Your account shows a team called "*Your Name* (Personal Team)".
3. Plug the iPhone into the Mac with a USB-C cable. Unlock it and tap
   **Trust** when it asks about the computer.
4. On the iPhone: **Settings > Privacy & Security > Developer Mode**, turn it
   on, and let the phone restart. Confirm when it asks. The switch only
   appears once the phone has been connected to a Mac running Xcode.

### 3. Sign with your Personal Team

Signing lives in a file git ignores, so your team ID never ends up in a
commit.

1. `open ios/Jetlink.xcodeproj`
2. Select the **Jetlink** project in the left sidebar, then the **Jetlink**
   target, then **Signing & Capabilities**. Under **Team**, choose your
   Personal Team.
3. Copy the team ID that choice wrote into the project, then undo the
   change:

   ```bash
   grep -m1 DEVELOPMENT_TEAM ios/Jetlink.xcodeproj/project.pbxproj
   git checkout ios/Jetlink.xcodeproj
   ```

4. Create `ios/Config/Local.xcconfig` with that ID and a bundle identifier of
   your own. The identifier must be unique to you; a reverse-domain name
   works:

   ```
   DEVELOPMENT_TEAM = ABCDE12345
   PRODUCT_BUNDLE_IDENTIFIER = com.yourname.jetlink
   ```

5. Back in Xcode, Signing & Capabilities should show your team and
   identifier, with no errors. If Xcode says your team does not support a
   capability, remove it with its trash icon. The app needs only
   **Increased Memory Limit**, which a free team can sign.

### 4. Install it on the iPhone

1. At the top of the Xcode window, pick your iPhone as the run destination.
2. Press **Run** (⌘R). Run builds Release, which is what the frame times
   need. The first build downloads onnxruntime (64 MB) and takes a few
   minutes.
3. The first launch is blocked until you trust your own certificate. On the
   iPhone: **Settings > General > VPN & Device Management**, tap your Apple
   ID, then **Trust**. Launch Jetlink again.
4. Allow **Local Network** access if iOS asks; the comma and the accuracy
   test reach the app over the network.

With a free account the installed app **stops opening after 7 days**. Plug
in and press Run again to renew it; the app keeps its models and settings.
Check the date before a drive. A free account can have at most 3 of its own
apps installed on a device at once.

### 5. Check it before going near the car

1. **Models** tab: download the model your comma drives with (the one picked
   under **Settings > Models > Big Model** on the comma), on Wi-Fi, and tap
   **Prepare**. On Automatic this also times the Neural Engine against the
   GPU.
2. **Status > Benchmark > Run for 1 minute.** The verdict should read "Fast
   enough": p99 at 35 ms or less, nothing over 50 ms. Then **Run for 10
   minutes** with the phone as it will be in the car (charging, in its
   mount) to see how much it slows as it warms. **Share the results** gives
   a text report.
3. **Accuracy:** copy the command under "Accuracy, from your Mac" on the
   Benchmark screen, and run it in this checkout on a Mac on the same Wi-Fi,
   with the model's `.onnx` file. It must end with `OK`.

## Connect it to the comma

**One cable, through a hub** (the comma stays up this way):

```
iPhone -> USB-C hub -> USB-A to USB-C cable -> comma
```

The comma presents itself as a USB network adapter, which iOS drives
itself. Plugged straight into the phone, the comma tries to power it and
reboots. Through the hub's USB-A port it never does, because a USB-A port
can only supply power. A hub with power pass-through lets the phone charge.
This route has kept the comma up; the link over it is not yet measured.

With the zoompilot patch applied ([below](#zoompilot-jetson-trt-limitations-and-the-patch)),
over SSH on the comma:

1. Copy the script over from this checkout, once:
   `scp ios/comma/setup_net_gadget.sh comma@<comma's address>:/data/`
2. Point the comma at the phone, once (it persists):

   ```bash
   echo -n 192.168.60.2:5599 > /data/params/d/JetlinkEndpoint
   ```

3. Turn **Settings > Models > Accelerator Link** off.
4. After every boot, with the cable plugged in:

   ```bash
   sudo bash /data/setup_net_gadget.sh
   ```

   It releases jetlink's own USB gadget, which zoompilot sets up at boot,
   and presents the network adapter with the comma at `192.168.60.1`.
5. On the iPhone: **Settings > Ethernet**, tap the new adapter, **Configure
   IP > Manual**: address `192.168.60.2`, subnet mask `255.255.255.0`, Router
   empty. iOS remembers this for the adapter. Check from the comma with
   `ping -c 3 192.168.60.2`.
6. Measure the link, still with Accelerator Link off: the Benchmark
   screen's "Over the link, from the comma" command.
7. Turn **Accelerator Link** back on. The app's Status tab shows **Comma:
   Connected**, and the comma's home-button icon pulses, then turns green.

**Or Ethernet:** a USB-C Ethernet adapter on the comma (Realtek
RTL8152/8153 or ASIX AX88179), a hub with Ethernet on the iPhone, and a
network cable. Instead of step 4, give the comma's adapter the address,
with its name from `ip link`:
`sudo ip addr add 192.168.60.1/24 dev <adapter> && sudo ip link set <adapter> up`.

Keep Jetlink open and on screen while driving: iOS suspends background apps.
The app keeps the screen awake while it serves.

To measure the link itself, parked:
- The Benchmark screen's "Over the link, from the comma" command runs
  `bench_link.py` on the comma. Turn Accelerator Link off for this test.
- zoompilot's own `tools/jetlink_live_bench.sh 180` runs the comma's real
  cameras and modeld, without controls, for three minutes.

## zoompilot jetson-trt: limitations and the patch

The comma side is zoompilot's [`jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
Its `JetlinkEndpoint` setting sends the link over TCP, which is what the
iPhone needs. That mode was written for bench bring-up with a Jetson on
Ethernet, and it has these limitations:

- **It never prepares anything while parked.** In TCP mode, zoompilot's
  gadget owner (`owner.py`) waits for a USB gadget that TCP mode never
  creates, so it never starts its parked provisioning run. That run:
  - records the server's engine as ready,
  - drives the home-button icon,
  - compiles the comma's camera warp.

  Without the warp, modeld never switches to the large model. A comma that
  already compiled its warp while using a Mac or Jetson over USB can still
  switch during a drive. A fresh setup never does.
- **The large model never joins a drive.** Before modeld uses a new
  connection, it waits for a Jetson to finish attaching over USB
  (`wait_for_host`). Over TCP nothing attaches over USB, so every attempt
  waited 45 seconds, gave up and started over. modeld stayed on the small
  model, stuck in `joining`. This showed up in zoompilot's parked bench
  (`tools/jetlink_live_bench.sh`) with the phone on the link.
- **Every connection waits 8 seconds first.** Before connecting, zoompilot
  asks the gadget owner to lend it the USB endpoints (`lending.py`). Over
  TCP there are none, so it waits out the 8-second timeout every time.
- **The order of steps matters.** Setting `JetlinkEndpoint` while zoompilot
  already holds the USB port doesn't release it; turning Accelerator Link
  off and on does.
- **No USB network link at boot.** zoompilot's boot script sets up only its
  own USB gadget, which iPhones cannot use. The one-cable network link has
  to be started by hand after every reboot, and is blocked by the power
  problem above.

`ios/comma/zoompilot-tcp-provisioning.patch` fixes the first three (21
lines in `owner.py`, `lending.py` and `gadget.py`, plus tests). With it,
107 of zoompilot's backend, gadget, owner, lending and jetlinkd tests pass,
and its 7 new tests fail without it. The other 3 are warp tests that need
openpilot's hardware module, which the test machine did not have; they fail
the same way with or without the patch. It applies cleanly to `jetson-trt`
at `bcb49d7`.

If you applied the earlier version of the patch, undo it with the old
file, then apply the new one:

```bash
ssh comma@<comma-ip> 'cd /data/openpilot && git apply -R /data/zoompilot-tcp-provisioning.patch'
scp ios/comma/zoompilot-tcp-provisioning.patch comma@<comma-ip>:/data/
ssh comma@<comma-ip> 'cd /data/openpilot && git apply /data/zoompilot-tcp-provisioning.patch'
```

### Applying the patch on the comma

This changes code on the comma, so do it parked. Make sure you can undo it
(shown below).

1. Turn on SSH: **Settings > Network > Enable SSH**, and add your GitHub
   username under **SSH Keys**. Find the comma's IP address under Settings >
   Network.
2. From this checkout on your Mac, copy the patch over, check it applies,
   and apply it:

   ```bash
   scp ios/comma/zoompilot-tcp-provisioning.patch comma@<comma-ip>:/data/
   ssh comma@<comma-ip> 'cd /data/openpilot && git apply --check /data/zoompilot-tcp-provisioning.patch && git apply /data/zoompilot-tcp-provisioning.patch'
   ```

3. Stop the updater from resetting it. openpilot's updater resets local
   changes when it installs an update:

   ```bash
   ssh comma@<comma-ip> 'echo -n 1 > /data/params/d/DisableUpdates'
   ```

4. Reboot the comma so the patched code runs.

To undo it:

```bash
ssh comma@<comma-ip> 'cd /data/openpilot && git apply -R /data/zoompilot-tcp-provisioning.patch && rm /data/params/d/DisableUpdates'
```

To keep it for good, commit the patch to a zoompilot branch of your own and
install the comma from that branch, instead of applying it by hand.

---

# Jetlink (upstream)

Run openpilot's large driving models on a computer plugged into your comma.
The comma keeps the cameras and vehicle control. It sends prepared camera
images over USB, the other computer runs the model, and predictions come back
20 times per second.

Jetlink is experimental. It needs a zoompilot build with Jetlink built in. The
comma side lives on the [zoompilot `jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
The small model you picked in sunnypilot keeps driving whenever the link is down. If the link
drops while engaged, the comma soft-disables and tells you to take over. See
[status and known limitations](docs/status.md).

<p align="center">
  <img src="docs/images/mac-status.webp" width="49%" alt="Jetlink for Mac: server status and model loading progress">
  <img src="docs/images/mac-models.webp" width="49%" alt="Jetlink for Mac: available models and download status">
</p>

## Quick start

Choose your computer: **[Mac](#mac)** · **[Jetson](#jetson)** · **[CUDA laptop](#cuda-laptop)**.
For an iPhone, see [Jetlink for iPhone](#jetlink-with-an-iphone-server) at the top of this page.
Then follow the shared [comma setup](#comma-setup-all-platforms).

You need a **comma 3X or comma 4**, a **USB 3 A-to-C data cable**, and
**separate power for both devices**. Charge-only cables will not work.
Keep the comma online and stay parked for the first setup.

### Mac

For **Apple silicon, macOS 15 or later**. 16 GB of memory is recommended.
Mac is bench-tested; Jetson is the tested in-car setup.

1. Download the **Mac ZIP** from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Double-click the ZIP to unzip it, then drag **Jetlink.app** to **Applications**.
3. Open **Jetlink**. The server starts automatically; **Waiting for comma**
   means it is ready to connect. Keep your Mac powered and awake.
4. Follow [comma setup](#comma-setup-all-platforms) below.

If macOS blocks an unsigned build, follow the release notes or the
[Mac install guide](docs/macos-app.md#if-the-build-is-not-signed).
No Python or Homebrew is needed for the app.

<details>
<summary>Developers: run from source</summary>

With Homebrew installed, run in Terminal:

```bash
brew install python libusb
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
scripts/run-mac.sh
```

The script installs its dependencies on first run. To build or work on the
GUI, see [macOS development](macos/README.md).

</details>

See the [Mac guide](docs/macos-app.md) for model downloads, settings, and logs.

### Jetson

For **Jetson Orin Nano Super (8 GB) with JetPack 6.2**. Use a power supply
sized for 25 W mode and allow several GB of free space on `/mnt/data`.

Run these commands in a terminal on the Jetson:

```bash
sudo apt update
sudo apt install -y git docker.io nvidia-container
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

For a **release install**, copy the Jetson image reference from the
[release notes](https://github.com/zoompilot/jetlink/releases). Set `IMAGE` to
that reference, replacing `VERSION` below with its version:

```bash
IMAGE=ghcr.io/zoompilot/jetlink:VERSION-jetson
sudo docker pull "$IMAGE"
sudo env IMAGE="$IMAGE" docker/run.sh --transport usb
```

If the release has no Jetson image, or you want to **build from source**, run
these commands instead from the `jetlink` folder:

```bash
sudo docker/build.sh
sudo docker/run.sh --transport usb
```

Leave the terminal open and follow [comma setup](#comma-setup-all-platforms).
Once it works, use the [Jetson guide](docs/jetson.md#start-at-boot) to start
Jetlink automatically at boot. For a release image, use your image reference
in place of `jetlink:latest` in that guide's image-inspection command.

### CUDA laptop

For a **Linux laptop with an NVIDIA GPU**, a working NVIDIA driver, and
**Python 3.10 or later**. This setup is hardware-tested.

On Ubuntu or Debian, run in a terminal:

```bash
sudo apt update
sudo apt install -y git python3-venv libusb-1.0-0
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[trt,usb,nvml]"
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
jetlink-server --backend trt --transport usb
```

Leave the terminal open, keep the laptop powered and awake, then follow
[comma setup](#comma-setup-all-platforms). If the comma was already plugged
in, unplug and reconnect it so the USB permissions take effect.

Next time, open a terminal in `jetlink` and run:

```bash
source .venv/bin/activate
jetlink-server --backend trt --transport usb
```

For **Docker** or **Windows with WSL2**, see the
[platform guide](docs/platforms.md#docker-nvidia-laptops-and-desktops).
Windows USB setup is not validated; start with the guide's TCP bench test.

## Comma setup (all platforms)

Do this once, whichever computer you chose.

1. **Install the Jetlink branch.** On a comma already running zoompilot, open
   **Settings > Software > Target Branch > Non-Prebuilt Branches** and select
   **jetson-trt**. Let it update, reboot, and finish building. Coming from
   another fork? Start with [zoompilot's installation instructions](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
2. **Enable Jetlink.** Open **Settings > Models** and turn on
   **Accelerator Link**. Leave **Big Model** on the default for your first run.
3. **Connect USB.** With Jetlink running on your computer, connect its
   **USB-A port to the comma's USB-C port** using a USB 3 data cable.
   On a Mac, use a USB-A port on a hub, dock, or USB-C-to-A adapter.
   On Jetson, use a USB-A port, not its USB-C port. A plain C-to-C cable may
   not connect reliably.
4. **Wait for green.** Stay parked with the comma online. Its home-button
   icon pulses while the model downloads, transfers, and prepares, then
   turns green when ready. No manual model download or SSH setup is needed.

The default model takes about **3 minutes to prepare on Jetson**. On Mac,
preparation takes about **10 seconds**, with later loads around **2 seconds**;
download time is extra. The screenshots above show an earlier app build.

| Icon | Meaning |
| --- | --- |
| Pulsing | Downloading, transferring, or preparing the model. Keep waiting. |
| Green | Parked: ready. Driving: the large model is driving. |
| Green, dimmed | Driving: ready and waiting for a chance to switch. Stop with cruise off, or turn lateral control off. |
| Orange | Preparation failed. Read the alert on the home screen. |
| Back to normal a minute later | Only with `--sleep-after`: the comma let the link go so the computer can sleep. Otherwise it stays connected the whole time you are parked. |

## What to expect when driving

- The small model drives while the server starts. On a computer that powers up
  with the car, the large model is prepared 65 to 96 seconds later.
- It takes over only when nothing is steering: **at a stop with cruise off, or
  with lateral control off**. Until then the icon is dimmed and the comma says
  **Big Model Available** at every stop. Disengaging alone is not enough on a
  car with lateral control always on.
- A **Big Model Ready** chime means it has taken over.
- Picking a new model needs the comma online once, while parked, to download
  it. After that it is prepared wherever you are: drive off in the middle and
  the small model drives, the panel counts the preparation down, and the large
  model joins at the first chance to switch.
- **Big Model Lost** while engaged is a soft disable. Take over. The small model
  drives, and Jetlink reconnects and switches back at the next chance.

To stop using Jetlink, turn off **Settings > Models > Accelerator Link**.

## If something is wrong

| Problem | Try |
| --- | --- |
| No Accelerator Link toggle | Check the branch in Settings > Software. |
| Toggle is on, nothing happens | Read the setup alert on the home screen. |
| Big Model list is empty | Connect the comma to the internet and use Refresh Model List. |
| Server keeps waiting, icon never pulses | Check the server is running, use a USB-A port, try another USB 3 data cable. |
| Orange icon | Read the alert, check the comma's internet, then toggle Accelerator Link off and on. |
| Model drops out repeatedly | Check the cable, separate power supplies, and cooling. |

<details>
<summary>Mac: an example of a server error</summary>

The Status screen shows the failure and recent output. Open **Logs** for more
detail, then check the [Mac troubleshooting guide](docs/macos-app.md#troubleshooting).

![Jetlink for Mac showing a server failure and diagnostic output](docs/images/mac-error.webp)

</details>

The [Jetson guide](docs/jetson.md#troubleshooting) has more, including how to
collect logs when reporting a problem.

## More

- [Jetlink for Mac, the app](docs/macos-app.md)
- [Jetson setup, boot service, troubleshooting, logs](docs/jetson.md)
- [Mac, Linux, Windows, Docker, and testing without a comma](docs/platforms.md)
- [Models, the model CLI, and the control channel](docs/models.md)
- [Status, known limitations, and measured performance](docs/status.md)
- [Updates and rollback](docs/releasing.md)
- [Cables, networking, and power](docs/transport.md)
- [Backends and measurements, for developers](docs/backends.md)

## License

[MIT](LICENSE).
