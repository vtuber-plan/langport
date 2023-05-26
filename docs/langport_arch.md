Langport Distribution Architecture
===

LangPort's distributed cluster architecture consists of two main types of nodes: workers and gateways. Workers are responsible for receiving tasks distributed by gateways, processing them, and sending back the results. Gateways, on the other hand, provide external interfaces and handle load balancing.

What sets LangPort's worker nodes apart from other distributed clusters is that they do not have a centralized controller or master node. Each node contains global node information, and any worker can act as a cluster access point.

This decentralized architecture provides several benefits, including increased fault tolerance and scalability. In the event of a node failure, the cluster can continue to function without disruption. Additionally, the lack of a central controller allows for easier scaling of the cluster as new nodes can be added without the need for reconfiguration or disruption of existing nodes.

LangPort's distributed cluster architecture requires users to provide an address for a neighboring node when starting a worker node. This can be any online worker node in the existing cluster. When a worker node is started, it automatically contacts the provided neighboring node and registers itself to join the cluster.

This approach ensures that the worker nodes are aware of the state of the cluster and can communicate with other nodes to distribute tasks and maintain cluster coherence. It also simplifies the process of adding new nodes to the cluster, as new nodes can be initialized with the address of any existing node, and the cluster can dynamically adjust to changes in the network topology.

Overall, LangPort's approach to cluster initialization and management provides a flexible and robust platform for serving large language models.

## Node Types
Currently, Langport includes two types of nodes: gateways and workers. Gateways support both OpenAI and FauxPilot protocols, while workers support two types of tasks: embedding and generation.

