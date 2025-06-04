In this branch the IP address for Ray processes is extracted from the NIC adapter  which needs to be
specified on the command line.
When no NIC is specified on the command line the IP address is extrected from the host name which won't
work on Cray systems and in general on systems with separate adapters for cotrol plane and high performance
network.
