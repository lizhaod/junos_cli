#!/usr/bin/env python3

import csv
import concurrent.futures
from jnpr.junos import Device
from jnpr.junos.exception import ConnectError
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
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
logging.basicConfig(level=logging.INFO)
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
            "no-more"
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
    parser.add_argument('-d', '--device', 
                      help='Filter devices by regex pattern (case-insensitive)',
                      default='')
    parser.add_argument('-o', '--output',
                      help='Output file path (supports .json, .txt, or .csv formats)',
                      default='')
    return parser.parse_args()

def load_devices(device_filter=''):
    """
    Load device information from CSV file.
    
    Args:
        device_filter (str): Regex pattern for filtering device names. 
                           Supports AND (,) and OR (|) operations.
                           Example: 'sin,r00' matches devices containing both 'sin' and 'r00'
                                  'sin|hkg' matches devices containing either 'sin' or 'hkg'
    """
    try:
        devices = []
        with open('devices.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # If host is empty, use the hostname (name) instead
                if not row['host'].strip():
                    row['host'] = row['name']
                
                # Apply device filter if specified
                if device_filter:
                    device_name = row['name'].lower()
                    device_filter = device_filter.lower()
                    
                    # Split the filter into components
                    if ',' in device_filter:
                        # AND operation using comma
                        patterns = [f.strip() for f in device_filter.split(',')]
                        if all(re.search(pattern, device_name) for pattern in patterns):
                            devices.append(row)
                    elif '|' in device_filter:
                        # OR operation using pipe
                        # Combine patterns with | for a single regex OR operation
                        combined_pattern = '|'.join(f'(?:{f.strip()})' for f in device_filter.split('|'))
                        if re.search(combined_pattern, device_name):
                            devices.append(row)
                    else:
                        # Single filter case
                        if re.search(device_filter, device_name):
                            devices.append(row)
                else:
                    devices.append(row)
        
        if not devices:
            logger.warning(f"No devices found matching the filter: {device_filter}")
            sys.exit(1)
            
        return devices
    except re.error as e:
        logger.error(f"Invalid regex pattern in device filter: {str(e)}")
        sys.exit(1)
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
    
    # Split command and grep pattern if exists
    command_parts = command.split(' | grep ')
    base_command = command_parts[0]
    grep_pattern = command_parts[1] if len(command_parts) > 1 else None
    
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
        'ConnectionAttempts': '2',
        'GSSAPIAuthentication': 'no',
        'PreferredAuthentications': 'password,keyboard-interactive',
        'NumberOfPasswordPrompts': '3',
        'KexAlgorithms': '+diffie-hellman-group14-sha1,diffie-hellman-group-exchange-sha1',
        'Ciphers': '+aes128-cbc,aes192-cbc,aes256-cbc,3des-cbc',
        'HostKeyAlgorithms': '+ssh-rsa,ssh-dss',
        'PubkeyAcceptedKeyTypes': '+ssh-rsa,ssh-dss'
    }
    
    # Common device parameters
    device_params = {
        'host': device_info['host'],
        'user': username,
        'password': password,
        'gather_facts': False,  # Skip fact gathering for faster connection
        'normalize': True,
        'timeout': 10,  # Connection timeout in seconds
        'attempts': 1,   # Set to 1 as we'll handle retries manually
        'auto_probe': 30,  # Auto probe every 30 seconds
        'ssh_config': None,  # Using custom ssh_options instead
        'ssh_private_key_file': None,
        'ssh_options': ssh_config
    }
    
    def try_connection(port, max_retries=2):
        """Try to connect using specified port with retries.""" 
        last_error = None
        for attempt in range(max_retries):
            try:
                with suppress_junos_logs():
                    # Update device parameters with current port
                    dev_params = device_params.copy()
                    dev_params['port'] = port
                    
                    # Attempt connection
                    dev = Device(**dev_params)
                    
                    with dev:
                        # Execute command based on type
                        if base_command.startswith('show'):
                            result = dev.cli(base_command, warning=False)
                            

                            # Apply grep filter if specified
                            if grep_pattern:
                                filtered_lines = []
                                for line in result.split('\n'):
                                    if grep_pattern.lower() in line.lower():
                                        filtered_lines.append(line)
                                result = '\n'.join(filtered_lines)
                        else:
                            # For configuration commands
                            with dev.config(mode='exclusive') as cu:
                                cu.load(base_command, format='set')
                                cu.commit()
                            result = "Configuration committed successfully"
                            

                        # Add port indicator to device name
                        port_indicator = "NETCONF" if port == 830 else "SSH"
                        device_name = f"{device_info['name']}:{port_indicator}"
                        
                        return {
                            'device': device_name,
                            'status': 'success',
                            'output': result
                        }
                        
            except (ConnectError, Exception) as e:
                last_error = e
                continue
        
        return None

    # Try NETCONF port first (830) with 2 retries
    result = try_connection(830, max_retries=2)
    if result:
        return result
        
    # Fallback to SSH port (22) with 2 retries
    result = try_connection(22, max_retries=2)
    if result:
        return result

    # If both attempts fail, now we log the error
    error_msg = "Failed to connect on both NETCONF (830, 2 retries) and SSH (22, 2 retries) ports"
    logger.error(f"{error_msg} for {device_info['name']} ({device_info['host']})")
    return {
        'device': device_info['name'],
        'status': 'error',
        'output': error_msg
    }

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
                        # Update progress description to show current device
                        progress.update(task, 
                                     advance=1, 
                                     description=f"[cyan]Processing {result['device']} ({future_to_device[future]['host']})...")
                        results.append(result)
                        
                        # If there's an error in the result, display it
                        if result['status'] == 'error':
                            error_console.print(f"[red]{result['device']}: {result['output']}[/red]")
                            
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        results.append({
                            'device': device['name'],
                            'status': 'error',
                            'output': error_msg
                        })
                        error_console.print(f"[red]{device['name']}: {error_msg}[/red]")
                        progress.update(task, advance=1)
                        
        finally:
            console.show_cursor(True)
    
    # Add a newline after all processing is done
    print("")
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to a file based on the file extension.""" 
    # Get file extension
    _, ext = os.path.splitext(output_file.lower())
    
    try:
        if ext == '.json':
            # Save as JSON
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        elif ext == '.csv':
            # Save as CSV
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Device', 'Status', 'Output'])
                for result in results:
                    writer.writerow([result['device'], result['status'], result['output']])
        else:
            # Default to txt format
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"Device: {result['device']}\n")
                    f.write(f"Status: {result['status']}\n")
                    f.write("Output:\n")
                    f.write(result['output'])
                    f.write("\n" + "="*50 + "\n\n")
        
        console.print(f"\n[green]Results saved to {output_file}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error saving results to {output_file}: {str(e)}[/red]")

def display_results(results: List[Dict[str, Any]], output_file: str = None):
    """Display results in a formatted table and optionally save to file.""" 
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Device", style="cyan")
    table.add_column("Status", width=12)
    table.add_column("Output")

    for result in results:
        status_color = "green" if result['status'] == 'success' else "red"
        table.add_row(
            result['device'],
            f"[{status_color}]{result['status']}[/{status_color}]",
            result['output']
        )

    console.print(table)
    
    # Save results if output file is specified
    if output_file:
        save_results(results, output_file)

def confirm_devices(devices):
    """Display filtered devices and ask for confirmation.""" 
    console.print("\n[bold blue]Filtered Devices:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("No.", style="dim", justify="right")
    table.add_column("Device Name")
    
    for idx, device in enumerate(devices, 1):
        table.add_row(str(idx), device['name'])
    
    console.print(table)
    console.print(f"\nTotal devices: [bold green]{len(devices)}[/bold green]")
    
    while True:
        response = input("\nProceed with these devices? (y/n): ").lower().strip()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            console.print("[yellow]Please enter 'y' for yes or 'n' for no.[/yellow]")

def main():
    """Main function to run the CLI tool.""" 
    try:
        args = parse_arguments()
        devices = load_devices(args.device)
        
        if not devices:
            console.print("[red]No devices found or all were filtered out[/red]")
            sys.exit(1)
        
        # Get user confirmation for the filtered devices
        if not confirm_devices(devices):
            console.print("[yellow]Operation cancelled by user[/yellow]")
            sys.exit(0)
        
        credentials = get_credentials()
        
        console.print("\n[blue]Type 'exit' to end the application[/blue]")
        console.print("[blue]Use Tab for command completion and Arrow keys for history[/blue]\n")
        
        while True:
            command = get_command()
            
            if command.lower() == 'exit':
                break
            
            results = execute_commands_with_progress(devices, command, credentials)
            display_results(results, args.output)
            
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
