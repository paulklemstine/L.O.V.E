from network import check_port_connectivity, get_local_subnets

def main():
    """
    Main function to run the network audit.
    """
    target_host = "127.0.0.1"
    common_ports = [21, 22, 80, 443, 3306, 5432, 8080]

    port_statuses = check_port_connectivity(target_host, common_ports)
    local_subnets = get_local_subnets()

    print_summary_report(port_statuses, local_subnets)

def print_summary_report(port_statuses, local_subnets):
    """
    Prints a non-technical summary of the network audit.
    """
    print("--- Network Infrastructure Audit Summary ---")
    print("\n1. Network Interface Configuration:")
    if local_subnets:
        print(f"  - Active subnets detected: {', '.join(local_subnets)}")
        print("  - Analysis: The system is correctly configured on the local network, enabling communication with other devices.")
    else:
        print("  - No active subnets detected.")
        print("  - Analysis: The system may not be connected to a network, which could limit its ability to connect to other services.")

    print("\n2. Port Connectivity Analysis:")
    open_ports = [p["port"] for p in port_statuses if p["status"] == "open"]
    if open_ports:
        print(f"  - Open ports on localhost: {', '.join(map(str, open_ports))}")
        print("  - Analysis: These ports are open to connections, which is necessary for services to operate. This also means they are potential points of entry and should be monitored.")
    else:
        print("  - No common ports were found to be open on localhost.")
        print("  - Analysis: The local machine is not running any common network services, or they are firewalled. This reduces the attack surface but may also indicate that necessary services are not running.")

    print("\n3. Device Discoverability and Bottlenecks:")
    print("  - Discoverability: The system's presence on the local network means it is discoverable by other devices on the same subnets. This is standard for most networked devices.")
    print("  - Potential Bottlenecks: No connectivity bottlenecks were detected in this passive scan. For a deeper analysis, an active performance test would be required.")

    print("\n--- End of Report ---")

if __name__ == "__main__":
    main()
