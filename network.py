# Based on https://gitlab.com/rwth-itsec/neon-and-xenon

import argparse
import subprocess
import sys
from typing import List

network_settings = {
    "loopback": (None, None),
    "LAN": ("1Gbit", "0.5ms 0.03ms 5%"),
    "WAN": ("100Mbit", "50ms 3ms 25%"),
    "WANfast": ("100Mbit", "15ms 0.5ms 15%"),
}

def start_virtual_network_setting(container, n_parties, network_setting):
    if network_setting in network_settings:
        bandwidth, delay = network_settings[network_setting]
        start_virtual_network(container, n_parties, bandwidth, delay)
    else:
        raise Exception(f"Invalid network setting '{network_setting}'")

def start_virtual_network(container, n_parties, bandwidth, delay):
    #####
    # Bridge:
    #####
    commands = [
        # Create a bridge
        f'ip link add name neon{container}_bridge txqueuelen 10000 type bridge',

        # set bridge up
        f'ip link set neon{container}_bridge up',

        # Give bridge an IP
        f'ip addr add 172.{16 + container}.1.10/16 brd + dev neon{container}_bridge'
    ]
    # Commands of intereset
    # sysctl -w net.bridge.bridge-nf-call-arptables=0
    # sysctl -w net.bridge.bridge-nf-call-iptables=0
    # sysctl -w net.bridge.bridge-nf-call-ip6tables=0
    # sysctl -w net.bridge.bridge-nf-call-ip6tables=0
    # sysctl -w net.ipv4.icmp_ratelimit=0

    run_commands(commands)

    #####
    # Virtual Networks
    #####
    def func_setup_network(i):
        ip = get_ip(container, i)
        run_commands([
            # Create a new network namespace nsi
            f'ip netns add neon{container}_ns{i}',

            # Create a veth pair to tunnel data into the bridge. vethi is the nsi side, br-vethi the bridge side
            f'ip link add neon{container}_veth{i} type veth peer name neon{container}_br_v{i}',

            # Move the interface vethi into the namespace nsi
            f'ip link set neon{container}_veth{i} netns neon{container}_ns{i}',

            # Give the interface an ip depending on the value i
            f'ip netns exec neon{container}_ns{i} ip addr add {ip}/24 dev neon{container}_veth{i}',

            # Set bridge interface side of tunnel up from default namespace
            f'ip link set neon{container}_br_v{i} up',

            # Set the other side of tunnel up from nsi
            f'ip netns exec neon{container}_ns{i} ip link set neon{container}_veth{i} up',

            # Set bridge br1 as the master of our bridge side interface of the tunnel
            f'ip link set neon{container}_br_v{i} master neon{container}_bridge',

            # Start local host so party 0 can communicate with itself in neon
            f'ip netns exec neon{container}_ns{i} ip link set dev lo up',
        ])

    for i in range(n_parties):
        func_setup_network(i)

    def func_check_network(party_index):
        for j in range(n_parties):
            if party_index == j:
                continue

            run_commands([f'ip netns exec neon{container}_ns{party_index} ping -c 1 {get_ip(container, j)} > /dev/null'])

            run_commands([f'ip netns exec neon{container}_ns{j} ping -c 1 {get_ip(container, party_index)} > /dev/null'])

    for i in range(n_parties):
        func_check_network(i)

    # Limited neighbour table size might lead to problems if many parties run on the same machine.
    # Should not happen here, so we do not include the workaround. If needed, have a look at
    # check_and_set_neighbor_table_size in the original neon code.

    def ratelimit(i):
        run_commands([
            # Outgoing, including delay
            f'ip netns exec neon{container}_ns{i} /sbin/tc qdisc add dev neon{container}_veth{i} root handle 1:0 htb default 10',
            f'ip netns exec neon{container}_ns{i} /sbin/tc class add dev neon{container}_veth{i} parent 1:0 classid 1:10 htb rate {bandwidth} quantum 1500',
            f'ip netns exec neon{container}_ns{i} /sbin/tc qdisc add dev neon{container}_veth{i} parent 1:10 handle 10:0 netem delay {delay}',

            # Incoming = Outgoing from bridge, no delay as already added by sender
            f'/sbin/tc qdisc add dev neon{container}_br_v{i} root handle 1:0 htb default 10',
            f'/sbin/tc class add dev neon{container}_br_v{i} parent 1:0 classid 1:10 htb rate {bandwidth} quantum 1500',
        ])

    if bandwidth is not None and delay is not None:
        for i in range(n_parties):
            ratelimit(i)

def stop_virtual_network(container, n_parties):
    # ip link list
    # ip netns list
    commands = [f"ip link delete neon{container}_bridge"]
    commands += [f"ip netns delete neon{container}_ns{i}" for i in range(n_parties)]
    run_commands(commands)

def get_ip(container: int, namespace: int) -> str:
    """Returns the IP address of the given client."""
    return f"172.{16 + container}.{1 + namespace // 245}.{11 + namespace % 245}"

def run_commands(commands: List[str]) -> None:
    for command in commands:
        sudo_command = "sudo " + command

        print('+ Executing command "{}"'.format(sudo_command))
        subprocess.check_call(sudo_command, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["start", "stop"], help="Mode: start or stop")
    parser.add_argument("n_parties", type=int, help="Number of parties")
    parser.add_argument("network_setting", help="Network setting")
    parser.add_argument("container", type=int, help="Container ID")

    if args.mode == "start":
        start_virtual_network_setting(container, n_parties, network_setting)
    elif mode == "stop":
        stop_virtual_network(container, n_parties)
