# Junos Multi-Device CLI Interaction Tool

This tool allows you to execute commands on multiple Junos devices simultaneously and view their outputs in an organized manner.

## Features
- Connect to multiple Junos devices simultaneously
- Execute CLI commands across all connected devices
- Display results in a clear, formatted table with line dividers
- Support for both operational and configuration commands
- Secure connection handling with error management
- Interactive credential input (no stored passwords)
- Command output filtering using grep
- Save command outputs to files in multiple formats (JSON, CSV, TXT)
- Filter devices by site code
- Concurrent execution with proper error handling
- Real-time progress tracking with:
  - Overall progress bar
  - Individual device connection status
  - Estimated time remaining
  - Task completion spinner
- Powerful regex-based device filtering

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python junos_cli.py                    # Run with all devices
python junos_cli.py -o results.json    # Save results to JSON file
python junos_cli.py -d sin -o out.txt  # Filter devices and save output
```

### Command-line Options
- `-d, --device`: Filter devices using regex patterns (see Device Filtering section)
- `-o, --output`: Save results to a file (supports .json, .csv, or .txt formats)

### Command Features
1. Regular Commands:
```
show version
show interfaces
```

2. Using Grep Filter:
```
show interfaces | grep ge-0
show version | grep Model
```

### Output Formats
When using the `-o` option, the following formats are supported:
- `.json`: Full structured data (best for programmatic use)
- `.csv`: Comma-separated values (good for spreadsheet applications)
- `.txt`: Human-readable text format with clear separators

## Device Filtering

The tool supports powerful regex-based device filtering using the `-d` or `--device` argument. This allows you to filter devices using:

- Regular expressions
- AND operations (using comma)
- OR operations (using pipe)

All patterns are case-insensitive.

### Filter Examples

```bash
# Basic regex patterns
python junos_cli.py -d "r00[ab]"      # Matches devices containing r00a or r00b
python junos_cli.py -d "^nyc"         # Matches devices starting with 'nyc'
python junos_cli.py -d "00a$"         # Matches devices ending with '00a'

# AND operation (comma-separated)
python junos_cli.py -d "nyc,r00"      # Matches devices containing both 'nyc' AND 'r00'
python junos_cli.py -d "^nyc.*,bb"    # Matches devices starting with 'nyc' AND containing 'bb'

# OR operation (pipe-separated)
python junos_cli.py -d "nyc|lax"      # Matches devices containing either 'nyc' OR 'lax'
python junos_cli.py -d "^nyc|^lax"    # Matches devices starting with either 'nyc' OR 'lax'
```

### Interactive Confirmation

After filtering devices, the tool will display a list of matched devices and prompt for confirmation:

```
Filtered Devices:
┌─────┬──────────────────┐
│ No. │ Device Name      │
├─────┼──────────────────┤
│   1 │ NYC-router1-r00a │
│   2 │ LAX-router2-r00b │
└─────┴──────────────────┘

Total devices: 2

Proceed with these devices? (y/n):
```

## Configuration

Edit `devices.csv` to add your devices. The file should be in CSV format with the following columns:
- name: Device name for identification (can include site code)
- host: IP address or hostname (optional - if empty, the name will be used as the hostname)

Example `devices.csv`:
```csv
name,host
NYC-router1,192.168.1.1
LAX-router2,10.0.0.2
BOS-router1,
```

In this example:
- NYC-router1 and LAX-router2 will be accessed using their IP addresses
- BOS-router1 will be accessed using its hostname directly

This flexibility allows you to:
- Use IP addresses when DNS is not available or for specific routing requirements
- Use hostnames when DNS is properly configured or in environments with dynamic IP addressing
- Mix both approaches in the same configuration file

## Security Features
- Credentials are prompted interactively and never stored
- Password input is masked during entry
- Same credentials are used for all devices to simplify management
- SSH key-based authentication is supported (recommended for production environments)

## Output Examples

### Progress Display
During command execution, you'll see:
- Overall progress bar showing completion status
- Individual device status updates
- Estimated time remaining
- Spinning indicator for active tasks
- Color-coded device names and status indicators

### Terminal Output
- Displays results in a formatted table with line dividers
- Color-coded status indicators (green for success, red for errors)
- Device names highlighted in cyan for better readability

### File Output Examples
1. JSON Format (results.json):
```json
[
  {
    "device": "NYC-router1",
    "status": "success",
    "output": "Hostname: NYC-router1\nModel: MX240\nJunos: 20.4R3"
  }
]
```

2. Text Format (results.txt):
```
Device: NYC-router1
Status: success
Output:
Hostname: NYC-router1
Model: MX240
Junos: 20.4R3
==================================================
```

3. CSV Format (results.csv):
```csv
Device,Status,Output
NYC-router1,success,"Hostname: NYC-router1\nModel: MX240\nJunos: 20.4R3"

```
