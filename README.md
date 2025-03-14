# JunosCLI - A Multi-Device Junos Command Execution Tool

JunosCLI is a powerful Python utility for executing commands across multiple Juniper devices simultaneously. It offers rich features like command auto-completion, parallel execution, and integration with PolarNG device inventory.

## Features

- **Multi-Device Execution**: Run commands across multiple Juniper devices simultaneously
- **Smart Auto-Completion**: Tab-completion for Junos commands with context-aware suggestions
- **PolarNG Integration**: Pull device inventory directly from PolarNG using polarcall
- **Rich Output Formatting**: Clear, colorized output with sorting options
- **Flexible Export Options**: Save results as JSON, CSV, or text files
- **Connection Resilience**: Automatic connection retry and protocol fallback
- **Authentication Management**: Secure credential handling with retry capabilities
- **Advanced Pipe Commands**: Support for grep, egrep and other pipe modifiers
- **Batch Command Processing**: Execute multiple commands from an input file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/junos_cli.git
cd junos_cli
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

```
jnpr.junos>=2.6.0
rich>=10.0.0
prompt_toolkit>=3.0.0
requests>=2.25.1
```

## Configuration

### Device List

JunosCLI can work with devices defined in two ways:

1. **Local CSV file** (devices.csv) with the following format:
```csv
name,host,description
router1,192.168.1.1,Core Router
router2,192.168.1.2,Edge Router
```

2. **Integration with PolarNG** using the polarcall.py utility

## Usage

### Basic Usage

```bash
# Using local devices.csv file
python junos_cli.py

# Using PolarNG integration
python junos_cli.py -p

# Output to file
python junos_cli.py -o results.json

# Batch mode with input command file
python junos_cli.py -i commands.txt -o results.txt
```

### Command-line Arguments

- `-p, --polarcall`: Use polarcall.py to generate device list from PolarNG
- `-o, --output FILE`: Save output to file (supports .json, .txt, or .csv)
- `-s, --sort`: Sort output (devices with output first, then alphabetically)
- `-i, --inputfile FILE`: Input file containing commands to be executed (one command per line)

### Interactive Mode

Once launched, JunosCLI provides an interactive shell with:

- Command history (arrow up/down)
- Tab completion for Junos commands
- Type 'exit' to quit

### Batch Mode

When using the `-i` flag with a command file, JunosCLI will:

1. Read commands from the specified file (one command per line)
2. Execute each command against all devices sequentially
3. Save all results to the output file specified with `-o`
4. Show progress for each command execution

Example commands.txt file:
```
show version
show interfaces terse
show system uptime
```

## Device Connection

JunosCLI attempts to connect to devices in the following order:

1. Try NETCONF connection on port 830
2. Fall back to SSH on port 22 if NETCONF fails
3. Apply multiple authentication retries with detailed error feedback

## Integration with PolarCall

When using the `-p` flag, JunosCLI will:

1. Look for polarcall.py in the expected locations
2. Allow you to specify filtering parameters for PolarNG
3. Generate a device list from PolarNG's inventory
4. Use the device information for command execution

## Command Execution

Commands are executed in parallel across all devices with:

- Progress indication for each device
- Detailed error reporting
- Timeouts to prevent hanging on unresponsive devices

## Output Options

Results can be:

- Displayed directly in the terminal (default)
- Sorted by success/output availability
- Exported to JSON, CSV, or text files

## Command Auto-Completion

JunosCLI includes completion for common Junos commands:

- Show commands (show version, show interfaces, etc.)
- Configuration commands
- Pipe modifiers (| display set, | match, etc.)

Press Tab to activate completion and see available options.

## Advanced Pipe Commands

JunosCLI supports advanced pipe commands for output filtering:

- `| grep pattern`: Filter lines containing pattern (supports standard grep options)
- `| egrep pattern`: Filter lines using regular expressions
- `| match pattern`: Simple string matching
- `| except pattern`: Exclude lines containing pattern
- `| count`: Count the number of lines
- `| find pattern`: Display from the first occurrence of pattern
- `| last N`: Display the last N lines

Example:
```
show interfaces terse | grep ge-
show bgp summary | egrep "Establ|Active"
show configuration | match interface
```

## Security Notes

- Device credentials are not stored on disk, only in memory during execution
- SSH strict host key checking is disabled by default
- Connection attempts are logged with appropriate verbosity

## Example Workflows

### Basic device check

```bash
# Launch the tool
python junos_cli.py

# When prompted, enter commands like:
show version
show interfaces terse
show system uptime
```

### Filtering devices with PolarCall

```bash
# Launch with PolarCall integration
python junos_cli.py -p

# When prompted for PolarCall parameters, enter filtering options:
-v "juniper" -m "mx" -n "core"
```

### Exporting JSON results

```bash
python junos_cli.py -o results.json -s
```

### Batch command execution

```bash
# Create a file with commands
echo "show version" > commands.txt
echo "show interfaces terse" >> commands.txt
echo "show system uptime" >> commands.txt

# Run in batch mode
python junos_cli.py -i commands.txt -o results.txt
```

## Troubleshooting

- If connections fail, check network connectivity and credentials
- For authentication errors, you'll be prompted to retry credentials
- Connection errors are logged with detailed information
- For PolarCall integration issues, ensure polarcall.py is correctly configured

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.