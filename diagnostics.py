import time
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

from utils import get_cpu_usage, get_memory_usage, get_disk_usage, get_process_list, parse_ps_output
from network import NetworkManager

def get_system_diagnostics(network_manager):
    """
    Aggregates various system metrics into a single dictionary.
    """
    diagnostics = {}

    # --- System Metrics ---
    cpu_data, cpu_error = get_cpu_usage()
    diagnostics['cpu'] = cpu_data if cpu_data else {'error': cpu_error}

    mem_data, mem_error = get_memory_usage()
    diagnostics['memory'] = mem_data if mem_data else {'error': mem_error}

    disk_data, disk_error = get_disk_usage()
    diagnostics['disk'] = disk_data if disk_data else {'error': disk_error}

    # --- Process Information ---
    proc_list, proc_error = get_process_list()
    if proc_list:
        parsed_procs = parse_ps_output(proc_list)
        diagnostics['processes'] = {
            'total_running': len(parsed_procs),
            'top_5_cpu': sorted(parsed_procs, key=lambda p: float(p.get('%cpu', 0)), reverse=True)[:5],
            'top_5_mem': sorted(parsed_procs, key=lambda p: float(p.get('%mem', 0)), reverse=True)[:5]
        }
    else:
        diagnostics['processes'] = {'error': proc_error}

    # --- Network Status ---
    if network_manager:
        diagnostics['network'] = {
            'p2p_online': network_manager.online,
            'peer_id': network_manager.peer_id,
            'connected_peers': len(network_manager.connections)
        }
    else:
        diagnostics['network'] = {'error': 'NetworkManager not available.'}

    return diagnostics

def format_diagnostics_panel(diagnostics):
    """
    Creates a rich Panel to display the formatted diagnostic information.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(size=5, name="footer")
    )
    layout["main"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split_column(Layout(name="cpu_mem"), Layout(name="disk"))

    # --- Header ---
    header_text = Text("J.U.L.E.S. System Diagnostics", justify="center", style="bold magenta")
    layout["header"].update(header_text)

    # --- CPU and Memory ---
    cpu_text = Text("CPU Usage: ", style="bold")
    if 'error' in diagnostics.get('cpu', {}):
        cpu_text.append(f"Error - {diagnostics['cpu']['error']}", style="red")
    else:
        cpu_percent = diagnostics['cpu']['cpu_usage_percent']
        color = "green" if cpu_percent < 50 else "yellow" if cpu_percent < 85 else "red"
        cpu_text.append(f"{cpu_percent}%", style=color)

    mem_text = Text("\nRAM Usage: ", style="bold")
    if 'error' in diagnostics.get('memory', {}):
        mem_text.append(f"Error - {diagnostics['memory']['error']}", style="red")
    else:
        ram = diagnostics['memory']['memory']
        color = "green" if ram['ram_used_percent'] < 50 else "yellow" if ram['ram_used_percent'] < 85 else "red"
        mem_text.append(f"{ram['ram_used_mb']}/{ram['ram_total_mb']} MB ({ram['ram_used_percent']}%)", style=color)

    layout["cpu_mem"].update(Panel(cpu_text + mem_text, title="Core Resources"))

    # --- Disk ---
    disk_text = Text("Disk Usage (/): ", style="bold")
    if 'error' in diagnostics.get('disk', {}):
        disk_text.append(f"Error - {diagnostics['disk']['error']}", style="red")
    else:
        disk = diagnostics['disk']
        disk_text.append(f"{disk['used_space']} / {disk['total_space']} ({disk['used_percent']}%)")
    layout["disk"].update(Panel(disk_text, title="Storage"))


    # --- Processes ---
    proc_text = Text("")
    if 'error' in diagnostics.get('processes', {}):
         proc_text.append(f"Error - {diagnostics['processes']['error']}", style="red")
    else:
        procs = diagnostics['processes']
        proc_text.append(f"Total Running: {procs['total_running']}\n\n", style="bold")
        proc_text.append("Top 5 by CPU:\n", style="bold cyan")
        for p in procs['top_5_cpu']:
            proc_text.append(f"  - {p['%cpu']}% | {p['command'][:50]}\n")
        proc_text.append("\nTop 5 by Memory:\n", style="bold cyan")
        for p in procs['top_5_mem']:
            proc_text.append(f"  - {p['%mem']}% | {p['command'][:50]}\n")

    layout["right"].update(Panel(proc_text, title="Process Monitor"))

    # --- Footer (Network) ---
    net_text = Text("P2P Status: ", style="bold")
    if 'error' in diagnostics.get('network', {}):
        net_text.append(f"Error - {diagnostics['network']['error']}", style="red")
    else:
        net = diagnostics['network']
        status = "Online" if net['p2p_online'] else "Offline"
        color = "green" if net['p2p_online'] else "red"
        net_text.append(f"{status}", style=color)
        if net['p2p_online']:
            net_text.append(f" | Peer ID: {net['peer_id'][:12]}... | Connections: {net['connected_peers']}")

    layout["footer"].update(Panel(net_text, title="Network Bridge"))

    return layout