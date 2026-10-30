![CSI-only multi-person pose estimation architecture](docs/multiperson-csi-design/figures/multiperson-csi-architecture.png)

## ESP32-S3 firmware setup: 3 TX + 3 RX at 100 Hz

Use the existing [firmware in sel00000/csi-pose](https://github.com/sel00000/csi-pose/tree/02b1001590beea8e7d6daac57a4d8b4a345eb50f/firmware). All three transmitters share one TX image, and all three receivers share one RX image. Assign each board its ID over serial after flashing.

### 1. Get the firmware and tools

You need six ESP32-S3 boards, USB data connections, and an activated [ESP-IDF 5.5.4 environment](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32s3/get-started/index.html). The default connection is UART0 through the board's USB-to-UART interface, such as CH340, at **921600 baud, 8N1, with no flow control**.

```bash
git clone https://github.com/sel00000/csi-pose.git csi-pose-firmware
cd csi-pose-firmware
git checkout 02b1001590beea8e7d6daac57a4d8b4a345eb50f
python -m pip install pyserial
```

Run the remaining commands from this `csi-pose-firmware/` directory. The COM ports below are examples; replace them with your actual ports. On Linux, use the corresponding `/dev/ttyUSB*` or `/dev/ttyACM*` paths.

| Board label | Firmware project | Internal ID | Example port |
|---|---|---|---|
| TX 1 | `firmware/tx` | `tx0` / `SET_IDX 0` | `COM34` |
| TX 2 | `firmware/tx` | `tx1` / `SET_IDX 1` | `COM35` |
| TX 3 | `firmware/tx` | `tx2` / `SET_IDX 2` | `COM36` |
| RX 1 | `firmware/rx` | `rx0` / `SET_IDX 0` | `COM22` |
| RX 2 | `firmware/rx` | `rx1` / `SET_IDX 1` | `COM23` |
| RX 3 | `firmware/rx` | `rx2` / `SET_IDX 2` | `COM24` |

### 2. Build and flash all six boards

Build once for each role, then flash the corresponding image to its three boards:

```bash
idf.py -C firmware/tx set-target esp32s3
idf.py -C firmware/tx build
idf.py -C firmware/tx -p COM34 flash
idf.py -C firmware/tx -p COM35 flash
idf.py -C firmware/tx -p COM36 flash

idf.py -C firmware/rx set-target esp32s3
idf.py -C firmware/rx build
idf.py -C firmware/rx -p COM22 flash
idf.py -C firmware/rx -p COM23 flash
idf.py -C firmware/rx -p COM24 flash
```

If you use the ESP32-S3's native USB Serial/JTAG connector for commands and data, run `idf.py -C firmware/tx menuconfig` and `idf.py -C firmware/rx menuconfig` **after setting the target and before building**. Select **CSI Link → Serial backend → USB Serial/JTAG** in both projects.

### 3. Assign IDs and set a common channel

Close any serial monitors or capture processes first. The existing `tx_ctl.py` utility works with both TX and RX boards and uses 921600 baud by default. `HELLO` reports each board's role and MAC address so you can check the labels and port assignments.

```bash
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 COM22 COM23 COM24 --cmd STOP
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 COM22 COM23 COM24 --cmd HELLO

python host/tools/tx_ctl.py --ports COM34 COM22 --cmd "SET_IDX 0"
python host/tools/tx_ctl.py --ports COM35 COM23 --cmd "SET_IDX 1"
python host/tools/tx_ctl.py --ports COM36 COM24 --cmd "SET_IDX 2"

python host/tools/tx_ctl.py --ports COM34 COM35 COM36 COM22 COM23 COM24 --cmd "SET_CH 6"
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 COM22 COM23 COM24 --cmd STATUS
```

All six boards must use the same channel. Send `STOP` before changing it. Check the `OK` replies and confirm the IDs and channel in `STATUS`; IDs and channel settings survive a reboot.

Each receiver learns and stores a TX MAC address for each TX ID. If you replace a transmitter or reassign its ID, keep all TX boards stopped and clear the saved mappings on all receivers before restarting:

```bash
python host/tools/tx_ctl.py --ports COM22 COM23 COM24 --cmd CLEAR_MACS
```

### 4. Start the receivers, then transmit at 100 packets/s

Open three terminals on the same host, one for each receiver, and run one command per terminal from `csi-pose-firmware/`:

```bash
# Terminal 1: RX 1
python host/bridge/bridge.py --port COM22 --rx-id 0 --no-mqtt --raw-dir logs --auto-start

# Terminal 2: RX 2
python host/bridge/bridge.py --port COM23 --rx-id 1 --no-mqtt --raw-dir logs --auto-start

# Terminal 3: RX 3
python host/bridge/bridge.py --port COM24 --rx-id 2 --no-mqtt --raw-dir logs --auto-start
```

The bridge sends `START` to its receiver and saves the binary CSI stream as `CSIRAW01` `.rawlog` files under `logs/`. This mode does not need an MQTT broker. Keep each RX port open only in its bridge process.

In a fourth terminal, start the three transmitters:

```bash
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 --cmd "START rate=100"
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 --cmd STATUS
```

Each TX targets approximately **100 packets/s**, with deliberate ±15% interval jitter to reduce recurring collisions. RX boards record incoming packets; they do not have a separate 100 Hz sampling setting. With all three transmitters active, each receiver should see TX IDs 0, 1, and 2, giving nine RX–TX links in total. Check that the per-link frame counts increase in each bridge's status output. Actual reception rates depend on the radio link and host.

To finish, stop the transmitters and press `Ctrl+C` in each RX bridge:

```bash
python host/tools/tx_ctl.py --ports COM34 COM35 COM36 --cmd STOP
```

After a reboot, start the receivers and transmitters again. Each TX `START` resets its sequence counter, so record restarts when collecting a session.
