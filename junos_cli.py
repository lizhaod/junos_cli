#!/usr/bin/env python3

import csv
import concurrent.futures
from jnpr.junos import Device
from jnpr.junos.exception import ConnectError, ConnectAuthError
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich import print as rprint
import sys
import logging
from getpass import getpass
from contextlib import contextmanager
import argparse
import json
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
import re
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress all ncclient-related logs
for logger_name in ['ncclient.transport.ssh', 'ncclient.transport.session', 'ncclient.operations.rpc']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

console = Console()

# Junos command list for auto-completion
JUNOS_COMMANDS = [
    # Show commands
    "show version", "show chassis hardware", "show system uptime",
    "show interfaces terse", "show interfaces detail", "show interfaces extensive",
    "show configuration", "show configuration | display set",
    "show route summary", "show route", "show route table",
    "show bgp summary", "show bgp neighbor",
    "show ospf neighbor", "show ospf interface", "show ospf database",
    "show isis adjacency", "show isis interface", "show isis database",
    "show ldp neighbor", "show ldp interface", "show ldp database",
    "show mpls interface", "show mpls lsp",
    "show system processes", "show system memory",
    "show system storage", "show system users",
    "show security policies", "show security zones",
    
    # NTP commands
    "show ntp associations",
    "show ntp status",
    "show ntp peers",
    "show configuration system ntp",
    "show system ntp threshold",
    
    # LLDP commands
    "show lldp", "show lldp neighbors",
    "show lldp neighbors detail", "show lldp statistics",
    "show lldp local-information", "show lldp interface",
    
    # Interface commands
    "show interfaces diagnostics optics",
    "show interfaces queue",
    "show interfaces statistics",
    "show interfaces descriptions",
    
    # System commands
    "show system cores",
    "show system alarms",
    "show system services",
    "show system commit",
    
    # Configuration commands
    "show configuration interfaces",
    "show configuration protocols",
    "show configuration routing-instances",
    "show configuration routing-options",
    "show configuration policy-options",
    "show configuration firewall",
    "show configuration security",
    
    # Routing commands
    "show route protocol bgp",
    "show route protocol ospf",
    "show route protocol static",
    "show route protocol direct",
    
    # Protocol-specific commands
    "show bgp groups",
    "show bgp summary",
    "show ospf overview",
    "show isis overview",
    
    # Common parameters
    "extensive", "detail", "brief", "terse",
    "| match", "| count", "| display set", "| display xml",
    "| no-more", "| except", "| find", "| last"
]

class JunosCompleter(Completer):
    def __init__(self, commands):
        self.commands = commands
        # Break down commands into levels
        self.command_tree = {}
        for cmd in commands:
            parts = cmd.split()
            current = self.command_tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Add pipe commands as single words
        self.pipe_words = [
            "display",
            "match",
            "except",
            "find",
            "count",
            "compare",
            "last",
            "trim",
            "resolve",
            "save",
            "no-more",
            "grep"  # Add grep to pipe commands
        ]
        
    def get_next_level_completions(self, words):
        """Get completions for the next level based on current input."""
        current = self.command_tree
        # Navigate to current position in tree
        for word in words[:-1]:  # All but last word
            found = False
            # Case-insensitive search
            for key in current:
                if key.lower() == word.lower():
                    current = current[key]
                    found = True
                    break
            if not found:
                return []
        
        # Find matches for last word
        last_word = words[-1].lower() if words else ""
        matches = []
        for key in current:
            if key.lower().startswith(last_word):
                matches.append(key)
        return matches
    
    def get_completions(self, document, complete_event):
        word_before_cursor = document.text_before_cursor
        
        # Handle pipe commands
        if '|' in word_before_cursor:
            # Get text after the last pipe
            pipe_parts = word_before_cursor.split('|')
            current_part = pipe_parts[-1].strip()
            
            # Complete pipe word if there's partial input
            if current_part:
                for word in self.pipe_words:
                    if word.lower().startswith(current_part.lower()):
                        yield Completion(word, start_position=-len(current_part))
            return
        
        # Normal command completion (no pipe)
        words = word_before_cursor.split()
        
        if not words:
            # Show all top-level commands
            for cmd in self.command_tree:
                yield Completion(cmd, start_position=0)
            return
        
        # Get matches for current level
        matches = self.get_next_level_completions(words)
        
        if matches:
            # Calculate replacement range
            last_word = words[-1]
            for match in matches:
                yield Completion(
                    match,
                    start_position=-len(last_word)
                )

def get_command():
    """Get command with auto-completion support."""
    session = PromptSession(
        history=FileHistory('.junos_cli_history')
    )
    completer = JunosCompleter(JUNOS_COMMANDS)
    kb = KeyBindings()
    
    @kb.add('tab')
    def _(event):
        """Handle tab completion.""" 
        buffer = event.current_buffer
        words = buffer.text.split()
        
        # Get completions for current position
        completions = list(completer.get_completions(buffer.document, None))
        
        if len(completions) == 1:
            # Single completion - add the completed word with a space
            completion = completions[0]
            word_before_cursor = words[-1] if words else ""
            new_word = completion.text + " "  # Add space after completion
            
            # Replace the last word with the completion
            if words:
                words[-1] = new_word
            else:
                words = [new_word]
            
            # Update buffer with the new text
            buffer.text = ' '.join(words)
            buffer.cursor_position = len(buffer.text)
            
        elif len(completions) > 1:
            # Show multiple possibilities
            console.print("\n[blue]Possible completions:[/blue]")
            for comp in completions:
                console.print(f"[blue]  {comp.text}[/blue]")
            console.print()
    
    while True:
        try:
            command = session.prompt(
                HTML('<ansiyellow>Enter command (exit to end): </ansiyellow>'),
                completer=completer,
                key_bindings=kb,
                complete_while_typing=False  # Only complete on Tab
            ).strip()
            
            if not command:
                continue
                
            if command.lower() == 'exit':
                return command
            
            return command
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            return 'exit'

@contextmanager
def suppress_junos_logs():
    """Temporarily suppress Junos connection logs.""" 
    original_level = logging.getLogger('ncclient.transport.session').level
    logging.getLogger('ncclient.transport.session').setLevel(logging.ERROR)
    try:
        yield
    finally:
        logging.getLogger('ncclient.transport.session').setLevel(original_level)

class LogCapture:
    """Context manager to capture and store log messages.""" 
    def __init__(self):
        self.messages = []
        self.handler = None
        self.has_error = False

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.has_error = True
        self.messages.append(self.handler.format(record))

    def __enter__(self):
        # Create a handler that stores messages
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.handler.emit = self.emit
        
        # Remove existing handlers and add our capture handler
        logger.handlers = []
        logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore default logging
        logger.handlers = []
        logger.addHandler(logging.StreamHandler())
        
    def display_logs(self):
        """Display all captured log messages only if errors occurred.""" 
        if self.has_error:
            console.print("\n[bold red]Connection Errors:[/bold red]")
            for message in self.messages:
                if "ERROR" in message or "WARNING" in message:
                    console.print(message)

def parse_arguments():
    """Parse command line arguments.""" 
    parser = argparse.ArgumentParser(description='Junos Multi-Device CLI Tool')
    parser.add_argument('-p', '--polarcall', action='store_true',
                      help='Use polarcall.py to generate device list')
    parser.add_argument('-o', '--output',
                      help='Output file path (supports .json, .txt, or .csv formats)',
                      default='')
    parser.add_argument('-s', '--sort', action='store_true',
                      help='Sort output: devices with output first, then alphabetically')
    parser.add_argument('-i', '--inputfile',
                      help='Input file containing commands to be executed (one command per line)',
                      default='')
    return parser.parse_args()

def load_devices(use_polarcall=False):
    """
    Load device information from CSV file or from polarcall.py.
    """
    try:
        devices = []
        
        if use_polarcall:
            # Check if polarcall.py exists in the expected location
            polarcall_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'Polarcall', 'polarcall.py')
            
            if not os.path.exists(polarcall_path):
                console.print(f"[red]Error: Could not find polarcall.py at {polarcall_path}[/red]")
                console.print("[yellow]Looking for polarcall.py in current directory...[/yellow]")
                polarcall_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polarcall.py')
                
                if not os.path.exists(polarcall_path):
                    console.print("[red]Error: Could not find polarcall.py[/red]")
                    return []

            temp_csv_file = 'temp_devices.csv'
            
            # Get polarcall parameters quietly
            console.print("[blue]Running polarcall.py to get device list...[/blue]")
            polarcall_params = Prompt.ask("Parameters", default="")
            
            # Redirect stdout to null to hide polarcall output
            cmd = f"python {polarcall_path} --format csv --output {temp_csv_file} {polarcall_params} > /dev/null 2>&1"
            
            # Execute polarcall command silently
            ret = os.system(cmd)
            
            if ret != 0:
                console.print("[red]Error executing polarcall.py[/red]")
                return []
            
            # Read CSV file silently
            if os.path.exists(temp_csv_file):
                try:
                    with open(temp_csv_file, 'r') as file:
                        csv_reader = csv.DictReader(file)
                        for row in csv_reader:
                            manufacturer = row.get('manufacturer', '')
                            model = row.get('model', '')
                            description = f"{manufacturer} {model}".strip() or 'Unknown'
                            
                            device = {
                                'name': row.get('hostname', ''),
                                'host': row.get('ip_address', ''),
                                'description': description
                            }
                            
                            if device['name'] and device['host']:
                                devices.append(device)
                finally:
                    # Clean up temp file silently
                    try:
                        os.remove(temp_csv_file)
                    except:
                        pass
            else:
                console.print(f"[red]Error: {temp_csv_file} not found[/red]")
                return []
        else:
            # Original CSV loading logic
            with open('devices.csv', 'r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if not row['host'].strip():
                        row['host'] = row['name']
                    
                    if 'description' not in row and 'model' in row:
                        row['description'] = row['model']
                    elif 'description' not in row:
                        row['description'] = 'Unknown'
                        
                    devices.append(row)
        
        if not devices:
            logger.warning("No devices found")
            sys.exit(1)
            
        return devices
        
    except Exception as e:
        logger.error(f"Error loading devices configuration: {str(e)}")
        sys.exit(1)

def get_credentials():
    """Prompt for username and password.""" 
    console.print("\n[bold blue]Enter credentials for device access:[/bold blue]")
    username = Prompt.ask("[yellow]Username[/yellow]")
    password = getpass("Password: ")
    return username, password

def execute_command(device_info, command, credentials):
    """Execute command on a single device and return the result.""" 
    username, password = credentials
    
    # Split command and separate grep from other pipe commands
    base_command = command
    device_pipes = []
    grep_pipes = []
    
    if ' | ' in command:
        parts = command.split(' | ')
        base_command = parts[0]
        for p in parts[1:]:
            if p.startswith('grep ') or p.startswith('egrep '):
                grep_pipes.append(f"| {p}")
            else:
                device_pipes.append(f"| {p}")
        
        # Reconstruct command to send to device (including display commands)
        if device_pipes:
            base_command = f"{base_command} {' '.join(device_pipes)}"
    
    logger.debug(f"Executing CLI command on {device_info['name']}: {base_command}")
    logger.debug(f"Will process locally: {grep_pipes}")
    
    # Enhanced SSH connection parameters
    ssh_config = {
        'StrictHostKeyChecking': 'no',
        'UserKnownHostsFile': '/dev/null',
        'ServerAliveInterval': '30',
        'ServerAliveCountMax': '5',
        'TCPKeepAlive': 'yes',
        'ControlMaster': 'auto',
        'ControlPersist': '10m',
        'ConnectTimeout': '10',
        'ConnectionAttempts': '2'
    }
    
    # Common device parameters
    device_params = {
        'host': device_info['host'],
        'user': username,
        'password': password,
        'gather_facts': False,
        'normalize': True,
        'timeout': 10,
        'port': 22,  # Default to SSH port
        'ssh_config': None,
        'ssh_private_key_file': None,
        'ssh_options': ssh_config
    }
    
    def try_connection(max_retries=2):
        """Try to connect with retries."""
        auth_error = None
        other_error = None
        
        for attempt in range(max_retries):
            try:
                with suppress_junos_logs():
                    dev = Device(**device_params)
                    with dev:
                        # Execute command with any non-grep pipes
                        result = dev.cli(base_command, warning=False)
                        
                        # Convert result to string if it's not already
                        if not isinstance(result, str):
                            result = str(result)
                            
                        # Process grep commands locally
                        if grep_pipes:
                            result = process_pipe_commands(result, grep_pipes)
                        
                        return {
                            'device': device_info['name'],
                            'status': 'success',
                            'output': result
                        }
                        
            except ConnectAuthError as e:
                auth_error = e
                continue
            except Exception as e:
                other_error = e
                continue
        
        # If we get here, all retries failed
        if auth_error:
            return {
                'device': device_info['name'],
                'status': 'auth_error',
                'output': str(auth_error)
            }
        else:
            return {
                'device': device_info['name'],
                'status': 'error',
                'output': str(other_error) if other_error else 'Connection failed after all retries'
            }
    
    # Try connection with retries
    return try_connection(max_retries=2)

def process_pipe_commands(output, pipe_commands):
    """Process pipe commands on the output."""
    result = output
    
    for cmd in pipe_commands:
        cmd = cmd.strip()
        logger.debug(f"Processing pipe command: {cmd}")
        
        # Add enhanced grep/egrep support
        if cmd.startswith('| grep ') or cmd.startswith('| egrep '):
            # Extract the full grep/egrep command after the pipe
            grep_cmd = cmd[2:].strip()  # Remove "| " prefix
            parts = grep_cmd.split()
            grep_type = parts[0]  # "grep" or "egrep"
            
            # Parse grep/egrep options and pattern
            pattern = None
            case_sensitive = True  # Default is case-sensitive
            invert_match = False   # Default is normal match (not inverted)
            
            # Process options
            i = 1
            while i < len(parts):
                if parts[i].startswith('-'):
                    # Option flags
                    if 'i' in parts[i]:
                        case_sensitive = False
                    if 'v' in parts[i]:
                        invert_match = True
                    i += 1
                else:
                    # Found the pattern
                    pattern = ' '.join(parts[i:])
                    # Handle quoted patterns - remove outer quotes if present
                    if (pattern.startswith("'") and pattern.endswith("'")) or \
                       (pattern.startswith('"') and pattern.endswith('"')):
                        pattern = pattern[1:-1]
                    break
            
            if pattern:
                try:
                    filtered_lines = []
                    is_egrep = grep_type == "egrep"
                    
                    for line in result.split('\n'):
                        try:
                            match_found = False
                            
                            # Determine if this is an OR pattern (contains | outside character classes)
                            has_or_pattern = '|' in pattern and is_egrep
                            
                            # Use regex for egrep, OR patterns, or if pattern contains regex special chars
                            if is_egrep or has_or_pattern or any(c in pattern for c in '[.*+?^$(){}\\'):
                                # Apply case insensitivity flag if needed
                                flags = re.IGNORECASE if not case_sensitive else 0
                                match_found = bool(re.search(pattern, line, flags=flags))
                            else:
                                # Simple string matching for basic grep
                                if not case_sensitive:
                                    match_found = pattern.lower() in line.lower()
                                else:
                                    match_found = pattern in line
                            
                            # Add line based on match and inversion flag
                            if match_found != invert_match:  # XOR logic
                                filtered_lines.append(line)
                                
                        except re.error as e:
                            # Log the regex error but continue processing
                            logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
                            # Fallback to simple string matching
                            if not case_sensitive:
                                match_found = pattern.lower() in line.lower()
                            else:
                                match_found = pattern in line
                                
                            if match_found != invert_match:
                                filtered_lines.append(line)
                                
                    result = '\n'.join(filtered_lines)
                except Exception as e:
                    logger.error(f"Error processing grep/egrep: {str(e)}")
                    
        # Keep existing pipe command processing
        elif cmd.startswith('| match '):
            pattern = cmd[8:].strip()
            filtered_lines = [line for line in result.split('\n') if pattern in line]
            result = '\n'.join(filtered_lines)
            
        elif cmd.startswith('| count'):
            count = len(result.split('\n'))
            result = f"Count: {count} lines"
            
        elif cmd.startswith('| display '):
            # Skip for now - these are typically handled by the device
            continue
            
        elif cmd.startswith('| except '):
            pattern = cmd[9:].strip()  # Fixed typo in .strip()
            filtered_lines = []
            for line in result.split('\n'):
                if pattern not in line:
                    filtered_lines.append(line)
            result = '\n'.join(filtered_lines)
            
        elif cmd.startswith('| find '):
            pattern = cmd[7:].strip()
            display = False
            filtered_lines = []
            for line in result.split('\n'):
                if pattern in line:
                    display = True
                if display:
                    filtered_lines.append(line)
            result = '\n'.join(filtered_lines)
            
        elif cmd.startswith('| no-more'):
            # This is handled by the device itself
            continue
            
        elif cmd.startswith('| last '):
            try:
                n = int(cmd[7:].strip())
                lines = result.split('\n')
                if len(lines) > n:
                    result = '\n'.join(lines[-n:])
            except:
                pass
    
    return result

def execute_commands_with_progress(devices, command, credentials):
    """Execute commands on all devices with a progress bar."""
    results = []
    error_console = Console(stderr=True, highlight=False)
    
    # Clear screen and hide cursor
    console.clear()
    console.show_cursor(False)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=10,
        expand=True
    ) as progress:
        try:
            # Create single progress task
            task = progress.add_task(
                f"[cyan]Executing command on {len(devices)} devices...",
                total=len(devices)
            )
            
            # Print a newline after progress bar for error messages
            print("")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_device = {}
                
                # Submit all tasks
                for device in devices:
                    future = executor.submit(execute_command, device, command, credentials)
                    future_to_device[future] = device
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_device):
                    device = future_to_device[future]
                    try:
                        result = future.result()
                        
                        # Update progress description
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Processing {device['name']} ({device['host']})..."
                        )
                        
                        # Handle string result (error case)
                        if isinstance(result, str):
                            error_result = {
                                'device': device['name'],
                                'status': 'error',
                                'output': result
                            }
                            error_console.print(f"[red]{device['name']}: {result}[/red]")
                            results.append(error_result)
                            continue
                            
                        # Handle None result
                        if result is None:
                            error_result = {
                                'device': device['name'],
                                'status': 'error',
                                'output': 'Command execution failed'
                            }
                            error_console.print(f"[red]{device['name']}: Command execution failed[/red]")
                            results.append(error_result)
                            continue
                            
                        # Handle dictionary result (success or error case)
                        if isinstance(result, dict):
                            if result['status'] == 'error':
                                error_console.print(f"[red]{result['device']}: {result['output']}[/red]")
                            results.append(result)
                            continue
                            
                        # Handle unexpected result type
                        error_result = {
                            'device': device['name'],
                            'status': 'error',
                            'output': f'Unexpected result type: {type(result)}'
                        }
                        error_console.print(f"[red]{device['name']}: Unexpected result type: {type(result)}[/red]")
                        results.append(error_result)
                        
                    except Exception as e:
                        logger.exception(f"Error executing command on {device['name']}")
                        error_result = {
                            'device': device['name'],
                            'status': 'error',
                            'output': str(e)
                        }
                        error_console.print(f"[red]{device['name']}: {str(e)}[/red]")
                        results.append(error_result)
                        progress.update(task, advance=1)
                        
        finally:
            console.show_cursor(True)
    
    # Add a newline after all processing is done
    print("")
    return results

def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort results by output presence and then alphabetically."""
    def sort_key(result):
        # First key: 0 if has output, 1 if empty or error (for reverse sorting)
        has_output = 0 if result['output'].strip() and result['status'] == 'success' else 1
        # Second key: device name in lowercase for alphabetical sorting
        device_name = result['device'].lower()
        return (has_output, device_name)
    
    return sorted(results, key=sort_key)

def save_results(results: List[Dict[str, Any]], output_file: str, command: str = "", append: bool = True):
    """Save results to a file based on the file extension.""" 
    # Get file extension
    _, ext = os.path.splitext(output_file.lower())
    
    # Check if file exists to determine if we're appending
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0 and append
    
    # Get current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        if ext == '.json':
            # Handle JSON format (special handling for append)
            if file_exists:
                # Read existing data
                try:
                    with open(output_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Add command and timestamp to results
                    for result in results:
                        result['command'] = command
                        result['timestamp'] = timestamp
                        
                    # Append new results
                    if isinstance(existing_data, list):
                        existing_data.extend(results)
                    else:
                        existing_data = [existing_data] + results
                        
                    # Write back the combined data
                    with open(output_file, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                except json.JSONDecodeError:
                    # If there's an error reading the JSON, append as new data
                    # Add command and timestamp
                    for result in results:
                        result['command'] = command
                        result['timestamp'] = timestamp
                        
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
            else:
                # New file - just write
                # Add command and timestamp
                for result in results:
                    result['command'] = command
                    result['timestamp'] = timestamp
                    
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        elif ext == '.csv':
            # Handle CSV format
            import csv
            
            # Append mode with header management
            mode = 'a' if file_exists else 'w'
            with open(output_file, mode, newline='') as f:
                writer = csv.writer(f)
                # Only write header for new files
                if not file_exists:
                    writer.writerow(['Timestamp', 'Command', 'Device', 'Status', 'Output'])
                
                # Add command separator if appending
                if file_exists:
                    writer.writerow(['=' * 20, '=' * 40, '=' * 20, '=' * 20, '=' * 20])
                    writer.writerow([timestamp, f"Command: {command}", '', '', ''])
                    
                for result in results:
                    writer.writerow([timestamp, command, result['device'], result['status'], result['output']])
        else:
            # Default to txt format
            # Append mode
            mode = 'a' if file_exists else 'w'
            with open(output_file, mode) as f:
                # Add timestamp and command header
                f.write(f"\n\n{'='*50}\n")
                f.write(f"Command: {command}\n")
                f.write(f"Executed at: {timestamp}\n")
                f.write(f"{'='*50}\n\n")
                
                # Write results
                for result in results:
                    f.write(f"Device: {result['device']}\n")
                    f.write(f"Status: {result['status']}\n")
                    f.write("Output:\n")
                    f.write(result['output'])
                    f.write("\n" + "-"*50 + "\n\n")
        
        # Only show message if not in batch mode (otherwise too many messages)
        if not getattr(save_results, 'batch_mode', False):
            action = "appended to" if file_exists else "saved to"
            console.print(f"\n[green]Results {action} {output_file}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error saving results to {output_file}: {str(e)}[/red]")

def display_results(results, output_file=None, sort_output=False, command=""):
    """Display results in a formatted table and optionally save to file.""" 
    if sort_output:
        results = sort_results(results)
        
    table = Table(title="Command Results", show_lines=True)  # Added show_lines=True for grid lines
    table.add_column("Device", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Output", style="green", no_wrap=False)
    
    for result in results:
        # Get status color
        status_style = {
            'success': 'green',
            'error': 'red',
            'auth_error': 'yellow'
        }.get(result['status'], 'white')
        
        # Format output based on status
        output = str(result.get('output', ''))  # Ensure output is a string
        if not output:
            output = 'No output'
            
        # Add row to table with appropriate styling
        table.add_row(
            result['device'],
            f"[{status_style}]{result['status']}[/{status_style}]",
            output
        )
    
    # Print table
    console.print(table)
    
    # Save results if output file is specified
    if output_file:
        save_results(results, output_file, command)

def confirm_devices(devices):
    """Display filtered devices and ask for confirmation.""" 
    console.print("\n[bold blue]Selected Devices:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("No.", style="dim", justify="right")
    table.add_column("Device Name")
    table.add_column("IP Address", style="cyan")
    table.add_column("Description", style="green")
    
    for idx, device in enumerate(devices, 1):
        # Get description if available, otherwise use "Unknown"
        description = device.get('description', 'Unknown')
        table.add_row(str(idx), device['name'], device['host'], description)
    
    console.print(table)
    console.print(f"\nTotal devices: [bold green]{len(devices)}[/bold green]")
    
    while True:
        response = input("\nProceed with these devices? (y/n): ").lower().strip()
        if response == 'y':
            output_file = None
            input_file = None
            
            log_response = input("\nLog output to a file? (y/n): ").lower().strip()
            if log_response == 'y':
                output_file = Prompt.ask("\nEnter output file name", default="junos_cli_output.txt")
                
                # Ask for input command file
                cmd_file_response = input("\nUse an input file with commands? (y/n): ").lower().strip()
                if cmd_file_response == 'y':
                    input_file = Prompt.ask("\nEnter input command file name", default="commands.txt")
                    
                    # Verify input file exists
                    if not os.path.exists(input_file):
                        console.print(f"\n[red]Warning: Input file '{input_file}' does not exist![/red]")
                        if not Prompt.ask("\nContinue without command file?", default="y"):
                            return False, None, None
                        input_file = None
            
            return True, output_file, input_file
        elif response == 'n':
            return False, None, None
        else:
            console.print("[yellow]Please enter 'y' for yes or 'n' for no.[/yellow]")

def read_commands_from_file(input_file):
    """Read commands from input file, one command per line."""
    commands = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                cmd = line.strip()
                if cmd and not cmd.startswith('#'):  # Skip empty lines and comments
                    commands.append(cmd)
        return commands
    except Exception as e:
        console.print(f"\n[red]Error reading commands from file {input_file}: {str(e)}[/red]")
        return []

def batch_process_commands(devices, commands, credentials, output_file, sort_output=False):
    """Process multiple commands from a file against all devices."""
    console.print(f"\n[blue]Batch processing {len(commands)} commands against {len(devices)} devices...[/blue]")
    
    # Ensure output file is specified for batch mode
    if not output_file:
        console.print("[red]Error: Output file is required for batch processing mode.[/red]")
        return False
        
    try:
        first_command = True
        for i, command in enumerate(commands, 1):
            console.print(f"\n[cyan]Executing command {i}/{len(commands)}: {command}[/cyan]")
            results = execute_commands_with_progress(devices, command, credentials)
            
            # In batch mode, only append to file, no console display
            append_mode = not first_command
            save_results(results, output_file, command, append=append_mode)
            
            if first_command:
                first_command = False
                
        console.print(f"\n[green]Batch processing complete. All results saved to {output_file}[/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Error during batch processing: {str(e)}[/red]")
        return False

def test_authentication(device, max_retries=3):
    """Test authentication with a device, allowing credential retries."""
    for attempt in range(max_retries):
        # Get credentials
        credentials = get_credentials()
        if not credentials:
            return None, None
            
        console.print(f"\n[cyan]Testing authentication with {device['name']} (Attempt {attempt + 1}/{max_retries})...[/cyan]")
        test_result = execute_command(device, "show version", credentials)
        
        if test_result['status'] == 'auth_error':
            console.print(f"\n[red]Authentication failed: {test_result['output']}[/red]")
            if attempt < max_retries - 1:
                console.print("\n[yellow]Please try again with different credentials.[/yellow]")
                continue
            else:
                console.print("\n[red]Maximum authentication attempts reached.[/red]")
                return None, None
        elif test_result['status'] == 'error':
            console.print(f"\n[yellow]Warning: Test device connection failed: {test_result['output']}[/yellow]")
            if not Prompt.ask("Do you want to proceed with all devices?", default="y"):
                return None, None
            return credentials, test_result
        else:
            console.print(f"\n[green]Authentication successful with {device['name']}[/green]")
            return credentials, test_result
            
    return None, None

def main():
    """Main function to run the CLI tool.""" 
    try:
        args = parse_arguments()
        
        # Validate input/output file requirements
        if args.inputfile and not args.output:
            console.print("[red]Error: When using an input file (-i), an output file (-o) must also be specified.[/red]")
            sys.exit(1)
            
        devices = load_devices(args.polarcall)
        
        if not devices:
            console.print("[red]No devices found or all were filtered out[/red]")
            sys.exit(1)
        
        # Get user confirmation for the filtered devices
        proceed, output_file, input_file = confirm_devices(devices)
        if not proceed:
            console.print("[yellow]Operation cancelled by user[/yellow]")
            sys.exit(0)
            
        # Use command line argument if provided, otherwise use the prompted file
        output_file = args.output or output_file
        input_file = args.inputfile or input_file
        
        # Validate input/output file requirements again (after user prompts)
        if input_file and not output_file:
            console.print("[red]Error: When using an input file, an output file must also be specified.[/red]")
            sys.exit(1)
        
        credentials, test_result = test_authentication(devices[0])
        if not credentials:
            return
        
        # Batch mode - process commands from file
        if input_file:
            commands = read_commands_from_file(input_file)
            if not commands:
                console.print("[red]Error: No valid commands found in the input file.[/red]")
                sys.exit(1)
                
            # Set batch mode flag
            save_results.batch_mode = True
                
            # Process all commands in batch mode
            success = batch_process_commands(devices, commands, credentials, output_file, args.sort)
            if not success:
                console.print("[red]Batch processing failed.[/red]")
                sys.exit(1)
        else:
            # Interactive mode
            console.print("\n[blue]Type 'exit' to end the application[/blue]")
            console.print("[blue]Use Tab for command completion and Arrow keys for history[/blue]\n")
            
            # Flag to track if it's the first command (for append mode)
            first_command = True
            
            while True:
                command = get_command()
                
                if command.lower() == 'exit':
                    break
                
                results = execute_commands_with_progress(devices, command, credentials)
                display_results(results, output_file, args.sort, command)
                
                # After first command, set flag for append mode
                if first_command:
                    first_command = False
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)
    finally:
        # Clean up history file
        try:
            history_file = '.junos_cli_history'
            if os.path.exists(history_file):
                os.remove(history_file)
        except:
            pass  # Ignore any errors during cleanup

if __name__ == "__main__":
    main()

def connect_with_retry(host, user, passwd, retries=3, delay=5):
    for attempt in range(retries):
        try:
            dev = Device(host=host, user=user, passwd=passwd, timeout=60)
            dev.open()
            return dev
        except ConnectError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

# Example usage
# dev = connect_with_retry('hostname', 'username', 'password')