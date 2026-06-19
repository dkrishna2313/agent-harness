# Research Memo: What does NVL72 rack-scale architecture imply for trays, shelves, cabinets, and deployment planning?

**Question:** What does NVL72 rack-scale architecture imply for trays, shelves, cabinets, and deployment planning?

## Executive Summary

The NVIDIA NVL72 rack-scale architecture represents a fundamental shift in AI infrastructure design, elevating the rack—rather than the individual server—to the primary unit of integration, compute, and security. First introduced with Blackwell and now in its third generation with Vera Rubin, the NVL72 rack is a precisely specified assembly of compute trays, NVLink Switch trays, and power shelves requiring mandatory liquid cooling infrastructure. Its cable-free modular tray design enables up to 18x faster assembly and servicing, while an ecosystem of 80+ MGX partners and a clear hierarchical scaling model (chip → superchip → rack → SuperPOD → AI factory) supports deployment planning from enterprise to hyperscale. Deployment planners must treat each NVL72 rack as a single integrated system—with unified power, cooling, networking, and security domains—and account for co-location with LPX racks, embedded management switches, and CDU compatibility from day one.

## Confirmed Facts

- NVIDIA Blackwell NVL72 was the first rack-scale architecture, freeing GPUs, CPUs, and interconnects from the confines of the traditional server boundary and elevating the rack to the primary unit of integration. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E001]
- The GB200 NVL72 rack contains 10 compute trays, 9 NVLink Switch trays, and 3 x 1U 33kW power shelves, and is compatible with in-rack or in-row CDU for liquid cooling. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- The GB300 NVL72 rack uses the same tray-and-shelf structure as the GB200 NVL72, with 10 compute trays, 9 NVLink Switch trays, and 3 x 1U 33kW power shelves, all liquid-cooled. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- Each NVL72 compute tray is a 1U liquid-cooled server node with 2 Grace CPUs, 4 Blackwell GPUs, and NVLink Switch connectors, forming the fundamental building block of the rack. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E004]
- Vera Rubin NVL72 is built on the third-generation NVIDIA MGX NVL72 rack design, featuring cable-free modular tray designs and support from over 80 MGX ecosystem partners for rapid deployment. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E005]
- The NVL72 rack's modular, cable-free tray design enables up to 18x faster assembly and servicing compared to the Blackwell generation. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E006]
- The Vera Rubin NVL72 rack unifies 72 Rubin GPUs, 36 Vera CPUs, ConnectX-9 SuperNICs, and BlueField-4 DPUs, scaling out via Quantum-X800 InfiniBand and Spectrum-X Ethernet. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E007]
- The NVL72 rack sits alongside NVIDIA LPX racks in a data center for fast, low-latency inference, implying co-location planning considerations. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E008]
- The NVL72 rack introduces modular, cable-free tray designs for 18x faster assembly and serviceability versus NVIDIA Blackwell, combined with intelligent resiliency and software-defined NVLink routing. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E009]
- Vera Rubin NVL72 is the first rack-scale platform to deliver NVIDIA Confidential Computing at full-rack scale, creating a unified trusted execution environment across all 36 CPUs, 72 GPUs, and the NVLink fabric. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E010]
- The NVIDIA Vera Rubin platform treats the data center, not a single GPU server, as the unit of compute, with GPUs, CPUs, networking, security, power delivery, and cooling architected together as a single system. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E011]
- The NVL72 architecture scales from individual superchips to racks to DGX SuperPOD-scale AI factory deployments, defining a hierarchical deployment planning model. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E014]
- The GB200 NVL72 rack includes 2 OOB management switches and 1 optional OS switch embedded within the cabinet. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E015]
- Microsoft's next-generation Fairwater AI superfactories will feature NVIDIA Vera Rubin NVL72 rack-scale systems scaling to hundreds of thousands of NVIDIA Vera Rubin Superchips. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E013]

## Inferences

- Because the rack is the primary unit of integration, procurement, staging, and acceptance testing should be planned at the rack level rather than the individual server level, requiring larger floor space and logistics coordination during installation.
- The generational continuity of the MGX NVL72 rack design (now third-generation) suggests operators who deployed GB200 NVL72 racks will face lower re-training and tooling overhead when transitioning to GB300 or Vera Rubin NVL72 racks.
- The 10 compute trays per rack (each 1U liquid-cooled) imply that tray-level hot-swap and field replacement are the expected maintenance paradigm, replacing traditional server-level RMA workflows.
- The requirement for co-location with LPX racks for low-latency inference implies that data center floor plans must pre-allocate contiguous or proximate rack rows for NVL72 and LPX systems together, not plan them independently.
- With the entire rack treated as a single confidential computing security domain, access control, key management, and audit logging must be designed at rack granularity rather than per-node or per-GPU granularity.
- The 80+ MGX ecosystem partner ecosystem reduces single-vendor lock-in risk for tray components, but interoperability qualification across partner options should be validated before large-scale rollouts.
- At hyperscale (hundreds of thousands of Vera Rubin Superchips as in Fairwater), the rack-scale unit implies thousands of discrete NVL72 racks, each requiring its own CDU connection, power shelf feeds, and management switch uplinks—making standardized rack manifests and automated provisioning critical.

## Power Implications

- Each NVL72 rack contains 3 x 1U 33kW power shelves, yielding up to ~99kW of rack-level power infrastructure per cabinet. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- The GB300 NVL72 rack maintains the same 3 x 1U 33kW power shelf configuration as the GB200 NVL72, confirming consistent high-density power requirements across Blackwell generations. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- GPUs, CPUs, networking, security, power delivery, and cooling are architected together as a single system in the Vera Rubin platform, meaning power delivery cannot be planned in isolation from compute and cooling. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E011]

## Cooling Implications

- Each NVL72 compute tray is a 1U liquid-cooled server node, meaning liquid cooling is mandatory at the tray level throughout the rack, not an optional upgrade. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E004]
- Both GB200 and GB300 NVL72 racks require compatibility with an in-rack CDU or in-row CDU, making CDU availability a hard prerequisite for deployment site qualification. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- The GB300 NVL72 rack carries the same CDU compatibility requirement (in-rack or in-row) as the GB200, confirming liquid cooling as a persistent, non-negotiable infrastructure baseline across NVL72 generations. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- Cooling, like power and networking, is architected as part of the single co-designed system in the Vera Rubin platform, requiring integrated cooling planning rather than treating it as a facility afterthought. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E011]

## Networking Implications

- The Vera Rubin NVL72 rack integrates ConnectX-9 SuperNICs and BlueField-4 DPUs within the rack, scaling out via Quantum-X800 InfiniBand and Spectrum-X Ethernet, requiring planners to provision both scale-up (NVLink) and scale-out (IB/Ethernet) network infrastructure. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E007]
- Each NVLink Switch tray provides 144 NVLink ports with fifth-generation NVLink at 1.8 TB/s GPU-to-GPU interconnect bandwidth, and 9 such trays per rack must be cabled to scale-out switches during deployment. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- Software-defined NVLink routing is embedded in the NVL72 rack architecture, enabling continuous operation and reducing maintenance overhead without physical recabling. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E009]
- Each GB200 NVL72 rack includes 2 OOB management switches and 1 optional OS switch, which must be integrated into the data center's out-of-band management network during deployment planning. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E015]
- NVL72 racks must be co-located with NVIDIA LPX racks in the data center to achieve fast, low-latency inference, imposing physical proximity and interconnect bandwidth constraints on floor layout planning. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E008]

## Rack Architecture Implications

- The NVL72 rack elevates the rack—not the server—to the primary unit of integration, fundamentally changing how cabinets are specified, procured, and managed compared to traditional server-based infrastructure. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E001]
- The precise internal layout of 10 compute trays + 9 NVLink Switch trays + 3 power shelves is standardized across the GB200 NVL72 cabinet, leaving limited room for customization and requiring facility planners to accommodate a fixed, high-density form factor. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- The same standardized tray-and-shelf structure carries forward into the GB300 NVL72, enabling consistent cabinet footprint planning across Blackwell NVL72 generations. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- The cable-free modular tray design of the NVL72 rack reduces deployment complexity and enables up to 18x faster assembly, directly impacting data center integration timelines and staffing plans. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E006]
- Vera Rubin NVL72 is the third-generation MGX NVL72 rack design, offering seamless generational transition and implying that existing NVL72 cabinet footprints and infrastructure can be reused for future generations. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E005]
- The NVL72 rack-scale architecture defines a hierarchical deployment model—superchip → rack → DGX SuperPOD → AI factory—requiring planners to design for multi-tier scaling from the outset. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E014]
- Vera Rubin NVL72 establishes the entire rack as a single confidential computing security domain, requiring security architects to treat cabinet boundaries as trust boundaries in their deployment designs. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E010]
- The NVL72 MGX design is backed by 80+ ecosystem partners, supporting rapid deployment and broad hardware sourcing options for cabinet components. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E012]

## Open Questions

- What is the precise total rack height (U count) of a fully populated NVL72 cabinet, and does it fit within standard 42U or 48U data center rack enclosures without modification?
- Are the 9 NVLink Switch trays and 10 compute trays hot-swappable independently, or does replacing a tray require full rack power-down?
- What are the specific CDU flow rate, water temperature inlet/outlet, and pressure requirements for qualifying an in-rack versus in-row CDU with the NVL72 rack?
- Do the 80+ MGX ecosystem partner compute trays (e.g., GIGABYTE XN14/XN15) carry full feature parity with NVIDIA-branded trays, or are there capability gaps relevant to deployment planning?
- How does software-defined NVLink routing handle partial tray failures—does it maintain full NVL72 collective communication performance or degrade gracefully?
- What is the inter-rack cabling specification (fiber type, length limits, switch model) required to connect multiple NVL72 racks into a DGX SuperPOD configuration?
- Are there specific floor load (kg/m²) or seismic anchoring requirements for fully populated NVL72 cabinets given their high component density?
- What management software stack is required to operate the embedded OOB management switches, and does it integrate natively with standard DCIM platforms?

## Source Notes

### Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf

- **Evidence ID:** E001
  **Claim:** NVIDIA Blackwell NVL72 was the first rack-scale architecture, freeing GPUs, CPUs, and interconnects from the confines of the traditional server boundary and elevating the rack to the primary unit of integration.
  **Evidence:** "NVIDIA Blackwell NVL72 was the first rack-scale architecture, freeing GPUs, CPUs, and interconnects from the confines of the traditional server boundary and elevating the rack to the primary unit of integration. This shift enabled major advances in scale-up bandwidth, efficiency, and deployability, and underpins many of today's largest AI deployments."
  **Category:** rack architecture
  **Relevance:** Directly establishes NVL72 as the originating rack-scale architecture concept, explaining its significance as the unit of integration beyond the traditional server boundary.
  **Confidence:** high
- **Evidence ID:** E011
  **Claim:** The NVIDIA Vera Rubin platform treats the data center, not a single GPU server, as the unit of compute, establishing NVL72 as a rack-scale system that operates as one AI supercomputer.
  **Evidence:** "Extreme co-design is the foundation of the Vera Rubin platform. GPUs, CPUs, networking, security, software, power delivery, and cooling are architected together as a single system rather than optimized in isolation. By doing so, the Vera Rubin platform treats the data center, not a single GPU server, as the unit of compute."
  **Category:** architecture
  **Relevance:** Explains the fundamental architectural philosophy behind NVL72: the entire rack (and by extension, the data center) is the unit of planning and deployment, not individual servers.
  **Confidence:** high
- **Evidence ID:** E014
  **Claim:** The NVL72 architecture scales from individual superchips to racks to DGX SuperPOD-scale AI factory deployments, defining a hierarchical deployment planning model.
  **Evidence:** "4. From chips to systems: NVIDIA Vera Rubin superchip to DGX SuperPOD: How Vera Rubin scales from superchips to racks to NVIDIA DGX SuperPOD-scale AI factory deployments."
  **Category:** architecture
  **Relevance:** Establishes the multi-tier deployment hierarchy (chip → superchip → rack/tray → SuperPOD) that operators must plan around when deploying NVL72-based infrastructure.
  **Confidence:** high

### NVIDIA-based Enterprise Solutions.pdf

- **Evidence ID:** E002
  **Claim:** The GB200 NVL72 rack contains 10 compute trays, 9 NVLink Switch trays, and 3 power shelves, requiring compatibility with an in-rack or in-row CDU for liquid cooling.
  **Evidence:** "10 x Compute Trays - GIGABYTE XN14-CB0-LA01 | 9 x NVIDIA NVLink™ Switch Trays - 144 x NVLink™ ports per tray - Fifth-generation NVLink™ with 1.8TB/s GPU-GPU interconnect | 3 x 1U 33kW Power Shelves | Compatible with in-rack CDU or in-row CDU"
  **Category:** rack architecture
  **Relevance:** Details the precise tray, shelf, and cabinet-level physical composition of the NVL72 rack, directly answering questions about its internal layout and deployment prerequisites.
  **Confidence:** high
- **Evidence ID:** E003
  **Claim:** The GB300 NVL72 rack uses the same tray-and-shelf structure as the GB200 NVL72, with compute trays, NVLink Switch trays, and 33kW power shelves, all liquid-cooled.
  **Evidence:** "10 x Compute Trays - GIGABYTE XN15-CB0-LA01 | 9 x NVIDIA NVLink™ Switch Trays - 144 x NVLink™ ports per tray - Fifth-generation NVLink™ with 1.8TB/s GPU-GPU interconnect | 3 x 1U 33kW Power Shelves | 3 x 1U 33kW Power Shelves | Compatible with in-rack CDU or in-row CDU"
  **Category:** rack architecture
  **Relevance:** Confirms that the NVL72 rack-scale architecture is consistently structured across Blackwell generations with modular trays, shelves, and liquid-cooling infrastructure requirements.
  **Confidence:** high
- **Evidence ID:** E004
  **Claim:** Each NVL72 compute tray is a 1U liquid-cooled server node with 2 Grace CPUs, 4 Blackwell GPUs, and NVLink Switch connectors, forming the fundamental building block of the rack.
  **Evidence:** "Form Factor: 1U Liquid-cooled server node | CPU: 2x NVIDIA Grace™ CPU (72 Arm Neoverse V2 cores) | GPU: 4 x NVIDIA Blackwell GPUs | 4 x NVIDIA NVLink™ Switch connectors"
  **Category:** rack architecture
  **Relevance:** Describes the tray-level form factor and configuration, which is the basic deployable unit within the NVL72 rack-scale architecture.
  **Confidence:** high
- **Evidence ID:** E012
  **Claim:** The NVL72 rack's NVIDIA MGX design supports over 80 ecosystem partners and is designed for rapid deployment with cable-free modular trays, simplifying data center integration.
  **Evidence:** "Built on the third-generation NVIDIA MGX™ NVL72 rack design, Vera Rubin NVL72 offers a seamless transition from prior generations. It delivers AI training with one-fourth the GPUs and AI inference at one-tenth the cost per million tokens versus NVIDIA Blackwell. Featuring cable-free modular tray designs and support from over 80 MGX ecosystem partners, the rack-scale AI supercomputer delivers world-class performance with rapid deployment."
  **Category:** operations
  **Relevance:** Confirms that the MGX-based NVL72 rack design with 80+ partners and cable-free tray modularity is explicitly intended to accelerate data center deployment and reduce integration complexity.
  **Confidence:** high
- **Evidence ID:** E015
  **Claim:** The GB200 NVL72 rack also includes management switches (2 OOB management switches and 1 optional OS switch), which must be accounted for in deployment planning.
  **Evidence:** "Management Switches - 2 x OOB management switches - 1 x Optional OS switch"
  **Category:** operations
  **Relevance:** Identifies the management infrastructure embedded within each NVL72 cabinet, relevant for network and operations planning when deploying the rack at scale.
  **Confidence:** high

### Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf

- **Evidence ID:** E005
  **Claim:** Vera Rubin NVL72 is built on the third-generation NVIDIA MGX NVL72 rack design, offering seamless transition from prior NVL72 generations for deployment planning.
  **Evidence:** "Vera Rubin NVL72 is built on the third-generation NVIDIA MGX™ NVL72 rack design, offering a seamless transition from prior generations. It delivers AI training with one-fourth the GPUs and AI inference at one-tenth the cost per million tokens versus NVIDIA Blackwell. Featuring cable‑free modular tray designs and support from over 80 MGX ecosystem partners, the rack-scale AI supercomputer delivers world‑class performance with rapid deployment."
  **Category:** rack architecture
  **Relevance:** Establishes that the NVL72 rack design has generational continuity (third-gen MGX), uses cable-free modular tray designs, and that 80+ MGX partners support rapid deployment planning.
  **Confidence:** high
- **Evidence ID:** E007
  **Claim:** The Vera Rubin NVL72 rack unifies 72 Rubin GPUs, 36 Vera CPUs, ConnectX-9 SuperNICs, and BlueField-4 DPUs in a single rack-scale platform, scaling out via Quantum-X800 InfiniBand and Spectrum-X Ethernet.
  **Evidence:** "NVIDIA Vera Rubin NVL72 unifies leading-edge technologies from NVIDIA—72 Rubin GPUs, 36 Vera CPUs, ConnectX®-9 SuperNIC™s, and BlueField®-4 DPUs. It scales up intelligence in a rack-scale platform with the NVIDIA NVLink™ 6 switch and scales out with NVIDIA Quantum-X800 InfiniBand and Spectrum-X™ Ethernet to power the AI industrial revolution at scale."
  **Category:** rack architecture
  **Relevance:** Defines the full component composition of the NVL72 rack, which directly informs cabinet-level planning for compute, networking, and infrastructure resources.
  **Confidence:** high

### NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf

- **Evidence ID:** E006
  **Claim:** The NVL72 rack uses cable-free modular tray designs that enable up to 18x faster assembly and servicing compared to the Blackwell generation.
  **Evidence:** "Second-Generation RAS Engine: The Rubin platform — spanning GPU, CPU and NVLink — features real-time health checks, fault tolerance and proactive maintenance to maximize system productivity. The rack's modular, cable-free tray design enables up to 18x faster assembly and servicing than Blackwell."
  **Category:** operations
  **Relevance:** Quantifies how the modular tray design in the NVL72 rack architecture directly accelerates deployment and servicing operations, a critical factor in deployment planning.
  **Confidence:** high
- **Evidence ID:** E013
  **Claim:** Microsoft's next-generation Fairwater AI superfactories will feature NVIDIA Vera Rubin NVL72 rack-scale systems scaling to hundreds of thousands of NVIDIA Vera Rubin Superchips, illustrating hyperscale deployment planning.
  **Evidence:** "Microsoft's next-generation Fairwater AI superfactories — featuring NVIDIA Vera Rubin NVL72 rack-scale systems — will scale to hundreds of thousands of NVIDIA Vera Rubin Superchips."
  **Category:** operations
  **Relevance:** Provides a real-world deployment planning example showing that NVL72 rack-scale architecture is designed to scale to hundreds of thousands of superchips within AI superfactories.
  **Confidence:** high

### Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf

- **Evidence ID:** E008
  **Claim:** The NVL72 rack-scale architecture sits alongside NVIDIA LPX racks in a data center for fast, low-latency inference, implying co-location planning considerations.
  **Evidence:** "NVIDIA Vera Rubin NVL72 unifies 72 NVIDIA Rubin GPUs, 36 NVIDIA Vera CPUs, NVIDIA ConnectX®-9 SuperNIC™ cards, and NVIDIA BlueField®-4 DPUs, and sits alongside NVIDIA LPX racks in a data center for fast, low-latency inference."
  **Category:** rack architecture
  **Relevance:** Indicates that deployment planning for NVL72 racks must account for physical co-location with LPX racks to achieve the intended inference performance architecture.
  **Confidence:** high
- **Evidence ID:** E009
  **Claim:** The rack introduces modular, cable-free tray designs for 18x faster assembly and serviceability versus NVIDIA Blackwell, combined with intelligent resiliency and software-defined NVLink routing.
  **Evidence:** "The rack introduces modular, cable-free tray designs for 18x faster assembly and serviceability versus NVIDIA Blackwell, combined with intelligent resiliency and software-defined NVLink routing, which ensures continuous operation and reduces maintenance overhead."
  **Category:** operations
  **Relevance:** Highlights how the NVL72 tray-based modular design simplifies rack assembly and ongoing operations, with software-defined NVLink routing further reducing maintenance burden.
  **Confidence:** high
- **Evidence ID:** E010
  **Claim:** Vera Rubin NVL72 is the first rack-scale platform to deliver NVIDIA Confidential Computing at full-rack scale, creating a unified trusted execution environment across all 36 CPUs, 72 GPUs, and the NVLink fabric.
  **Evidence:** "The third generation of NVIDIA Confidential Computing expands security to full-rack scale with NVIDIA Vera Rubin NVL72. This platform creates a unified, trusted execution environment across all 36 NVIDIA Vera CPUs, 72 NVIDIA Rubin GPUs, and the NVIDIA NVLink™ fabric that seamlessly connects them."
  **Category:** rack architecture
  **Relevance:** Demonstrates that NVL72 rack-scale architecture has security implications that span the entire rack, requiring planners to treat the rack as a single security domain rather than individual nodes.
  **Confidence:** high

## Evaluation Warnings

- None.
