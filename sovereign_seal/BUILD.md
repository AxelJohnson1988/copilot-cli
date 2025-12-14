# Sovereign Seal - Build Instructions

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Building the Viewer Executable

PyInstaller bundles `viewer.py` into a single standalone executable. The `--add-data` flag syntax differs by OS.

### Windows

```bash
pyinstaller --onefile --name SovereignSealViewer --add-data "payload.seal;." viewer.py
```

**Note:** Windows uses `;` (semicolon) as the path separator.

### macOS / Linux

```bash
pyinstaller --onefile --name SovereignSealViewer --add-data "payload.seal:." viewer.py
```

**Note:** macOS and Linux use `:` (colon) as the path separator.

### Cross-Platform Build Script

For convenience, you can use this Python snippet to detect the OS:

```python
import platform
import subprocess

sep = ";" if platform.system() == "Windows" else ":"
cmd = f'pyinstaller --onefile --name SovereignSealViewer --add-data "payload.seal{sep}." viewer.py'
subprocess.run(cmd, shell=True)
```

## Build Options

| Option | Description |
|--------|-------------|
| `--onefile` | Bundle everything into a single executable |
| `--name` | Name of the output executable |
| `--add-data` | Include additional data files (src{sep}dest) |
| `--noconsole` | Hide console window (Windows GUI apps) |
| `--icon` | Custom icon for the executable |

### Recommended Production Build (Windows)

```bash
pyinstaller --onefile --noconsole --name SovereignSealViewer --icon seal.ico viewer.py
```

### Recommended Production Build (macOS/Linux)

```bash
pyinstaller --onefile --name SovereignSealViewer --icon seal.icns viewer.py
```

## Output

After building, the executable will be in the `dist/` directory:
- Windows: `dist/SovereignSealViewer.exe`
- macOS: `dist/SovereignSealViewer`
- Linux: `dist/SovereignSealViewer`

## Usage Workflow

1. **Encrypt your patent PDF:**
   ```bash
   python builder.py patent.pdf -o payload.seal
   ```

2. **Build the viewer executable:**
   ```bash
   # On Windows:
   pyinstaller --onefile --name SovereignSealViewer --add-data "payload.seal;." viewer.py

   # On macOS/Linux:
   pyinstaller --onefile --name SovereignSealViewer --add-data "payload.seal:." viewer.py
   ```

3. **Distribute:**
   - Send `dist/SovereignSealViewer` (or `.exe`) to the recipient
   - Share the password securely via separate channel

4. **View (recipient):**
   ```bash
   ./SovereignSealViewer
   # Enter password when prompted
   ```

## Security Notes

- The `payload.seal` file contains: `SS1` header + 16-byte salt + 12-byte nonce + AES-256-GCM ciphertext
- Key derivation uses Scrypt with parameters: n=2^15, r=8, p=1
- Decrypted PDF exists only in RAM, never written to disk
- Leakage guard prevents extraction of 30+ consecutive verbatim words
