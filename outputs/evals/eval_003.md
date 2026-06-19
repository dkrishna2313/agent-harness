# Research Memo: How do NVLink, InfiniBand, Ethernet, Spectrum, and ConnectX shape AI data center networking?

**Question:** How do NVLink, InfiniBand, Ethernet, Spectrum, and ConnectX shape AI data center networking?

## Executive Summary

NVIDIA's AI data center networking strategy is built on a two-tier, co-designed fabric: a scale-up layer anchored by NVLink (now in its 6th generation at 3.6 TB/s per GPU / 260 TB/s per rack) that unifies GPUs within a rack into a single performance and security domain, and a scale-out layer using Quantum-X800 InfiniBand or Spectrum-X Ethernet (with co-packaged silicon photonics) that connects racks and clusters spanning hundreds of thousands to millions of GPUs. ConnectX SuperNICs (generations 7 through 9) serve as the endpoint interface between GPU nodes and the scale-out fabric. Each successive chip generation—Spectrum-5/6/7, NVLink 5/6, ConnectX-7/8/9—is co-designed as part of a synchronized platform to slash training time, inference token cost, and data center power consumption.

## Confirmed Facts

- NVLink 6 provides 3.6 TB/s of GPU-to-GPU bandwidth per GPU and 260 TB/s of rack-scale connectivity in the Vera Rubin NVL72, doubling the bandwidth of Blackwell's NVLink 5. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E001]
- The NVLink fabric in the Vera Rubin NVL72 creates a unified, trusted execution environment spanning all 36 Vera CPUs, 72 Rubin GPUs, and the NVLink fabric for confidential computing at rack scale. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E002]
- The Vera Rubin NVL72 scales out across GPU clusters using NVIDIA Quantum-X800 InfiniBand and NVIDIA Spectrum-X Ethernet, complementing the NVLink scale-up fabric. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E003]
- The Vera Rubin platform includes six co-designed chips—among them the ConnectX-9 SuperNIC and the Spectrum-6 Ethernet switch—to form a synchronized scale-out networking fabric. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]
- NVLink 6 switch delivers 3.6 TB/s all-to-all scale-up bandwidth per GPU, enabling high-speed GPU-to-GPU communications for AI within the rack. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E005]
- ConnectX-9 SuperNICs deliver 1.6 Tb/s of per-GPU bandwidth with programmable RDMA for low-latency, GPU-direct networking at massive scale. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E006]
- Spectrum-X Ethernet co-packaged optics switches deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime over traditional networking with pluggable transceivers. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E007]
- The Rubin platform uses extreme co-design across six chips—including NVLink 6 Switch, ConnectX-9 SuperNIC, and Spectrum-6 Ethernet Switch—to slash training time and inference token generation cost. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E008]
- NVLink 6 delivers 3.6 TB/s per GPU and 260 TB/s rack-wide bandwidth, with in-network compute to accelerate collective operations for large MoE models. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E009]
- Spectrum-X brings AI performance to Ethernet and, combined with silicon photonics, enables connection of hundreds of thousands of GPUs while saving megawatts of power. [Source: 9d2c3468-1af8-4ed1-9b85-31bdebff03dc.pdf, Evidence: E010]
- Co-packaged silicon photonics networking switches allow AI factories to scale to millions of GPUs, providing 3.5x power efficiency and 10x higher resiliency versus pluggable optics, replacing 432 transceivers with 72 NVIDIA Photonics units. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E011]
- The NVIDIA AI platform roadmap progresses from Spectrum-5 (51T, CX8 800G) through Spectrum-6 (102T, CPO, CX9 1600G) to Spectrum-7 (204T, CPO, CX10), alongside NVLink generation upgrades. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E012]
- The Vera Rubin NVL72 scales out with Quantum-X800 InfiniBand and Spectrum-X Ethernet to sustain high GPU cluster utilization and reduce time-to-train and total cost of ownership. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E013]
- The GB300 NVL72 is built for AI reasoning and based on Quantum-X800 InfiniBand or Spectrum-X Ethernet paired with ConnectX-8 SuperNIC for scale-out networking. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E015]
- The GB200 NVL72 compute tray includes ConnectX-7 NICs and BlueField-3 DPUs for networking, while using NVLink Switch trays with 1.8 TB/s GPU-GPU interconnect for scale-up. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E016]
- The DGX B300 system is equipped with NVIDIA ConnectX-8 high-speed networking at 800 Gb/s for AI reasoning workloads. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E017]
- The NVIDIA AI platform chips purpose-built for AI supercomputing span GPU, CPU, DPU, NIC, NVLink Switch, IB Switch, and Ethernet Switch, all unified by cluster-scale software including NCCL. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E018]

## Inferences

- The two-tier fabric model (NVLink for scale-up, InfiniBand/Spectrum-X for scale-out) suggests that AI data center architects must plan for fundamentally different bandwidth regimes within a rack versus between racks—intra-rack at 260 TB/s aggregate vs. per-GPU scale-out at 1.6 Tb/s via ConnectX-9.
- The co-design of all six Rubin platform chips (including networking ASICs) implies that mixing third-party NICs, switches, or interconnects would likely sacrifice the performance and efficiency guarantees NVIDIA quotes, creating strong platform lock-in pressure.
- The generational ConnectX roadmap (CX7→CX8→CX9→CX10) doubling bandwidth each generation suggests that switch fabric and cabling plant upgrades will be required at each GPU generation refresh, increasing CapEx frequency.
- NVLink's expansion into a security domain boundary (Confidential Computing at rack scale) suggests future AI workloads requiring data isolation—e.g., multi-tenant inference—may demand full NVL72 racks as the minimum trust boundary, complicating shared-infrastructure deployments.
- The use of Spectrum-X as the inter-rack fabric between Rubin NVL72 and LPX racks for disaggregated inference (E014) suggests that Ethernet, not InfiniBand, may become the preferred fabric for heterogeneous, multi-vendor AI inference clusters.
- In-network compute within NVLink 6 (E009) for accelerating collective operations implies that all-reduce and all-to-all operations for large MoE models are increasingly offloaded from GPU compute cycles to the switch fabric, a paradigm shift from traditional HPC collectives.

## Power Implications

- Spectrum-X Ethernet co-packaged optics switches deliver 5x better power efficiency over traditional networking with pluggable transceivers, directly reducing data center power draw at the switch layer. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E007]
- Co-packaged silicon photonics networking switches provide 3.5x power efficiency improvement versus pluggable optics and replace 432 discrete transceivers with 72 NVIDIA Photonics units, substantially cutting optical interface power. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E011]
- Spectrum-X with silicon photonics enables connection of hundreds of thousands of GPUs while saving megawatts of power at data center scale. [Source: 9d2c3468-1af8-4ed1-9b85-31bdebff03dc.pdf, Evidence: E010]

## Cooling Implications

- The extreme bandwidth density of NVLink 6 (3.6 TB/s per GPU, 260 TB/s per rack) concentrated within the NVL72 rack implies very high thermal density within the scale-up switch fabric, reinforcing the need for liquid cooling at rack scale. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E001]
- The GB300 NVL72 is described as a fully liquid-cooled, rack-scale design, indicating that the high-density NVLink and ConnectX networking components within the rack require liquid cooling infrastructure. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E015]
- Co-packaged silicon photonics integration (replacing 432 pluggable transceivers with 72 NVIDIA Photonics modules) reduces optical transceiver heat load per switch, potentially easing thermal management in scale-out switching rows. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E011]

## Networking Implications

- NVLink 6's 3.6 TB/s per-GPU / 260 TB/s rack-scale bandwidth with in-network compute enables support for massive Mixture-of-Experts (MoE) model training and inference without scale-out bottlenecks within a single rack domain. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E009]
- Quantum-X800 InfiniBand and Spectrum-X Ethernet serve as alternative but complementary scale-out fabrics, giving data center operators a choice between InfiniBand (traditional HPC/AI training) and Ethernet (broader ecosystem) for inter-rack connectivity. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E015]
- ConnectX-9 SuperNICs provide 1.6 Tb/s per-GPU bandwidth with programmable RDMA and GPU-direct capability, doubling the 800 Gb/s of ConnectX-8 in the prior Blackwell Ultra generation. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E006]
- Spectrum-X co-packaged optics switches deliver 10x higher network resiliency and up to 5x more uptime versus pluggable transceiver-based switches, improving AI cluster availability. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E007]
- When LPX and Vera Rubin NVL72 racks are co-deployed, they are connected via a custom Spectrum-X-based interconnect for cooperative decode token processing, indicating Spectrum-X serves as a disaggregated inference inter-rack fabric. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E014]
- All scale-up and scale-out networking chips across the Rubin platform are co-designed and unified by cluster-scale software including NCCL, CUDA, and DOCA, making the software stack an integral part of the networking architecture. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E018]
- The Spectrum roadmap progresses to 204 Tb/s aggregate switch capacity (Spectrum-7) with co-packaged optics and ConnectX-10 at the endpoint, indicating that scale-out bandwidth will continue to roughly double each generation. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E012]

## Rack Architecture Implications

- The Vera Rubin NVL72 integrates 72 GPUs, 36 CPUs, ConnectX-9 SuperNICs, BlueField-4 DPUs, and NVLink 6 Switches within a single rack, making the rack—not the node—the fundamental unit of AI compute and networking. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E003]
- NVLink's role as a security domain boundary for Confidential Computing means the NVL72 rack must be treated as an atomic, trusted unit, constraining workload placement and multi-tenant sharing at the rack level. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E002]
- The GB200 NVL72 rack architecture uses dedicated NVLink Switch Trays (separate from compute trays) with 144 NVLink ports per tray for scale-up, indicating that NVLink switching consumes dedicated rack real estate distinct from GPU compute trays. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E016]
- Co-deployment of Vera Rubin NVL72 racks alongside LPX racks connected via Spectrum-X requires data center planners to design for heterogeneous rack types with an inter-rack Ethernet fabric, not a homogeneous GPU cluster topology. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E014]
- The six co-designed chips of the Rubin platform—including NVLink 6 Switch, ConnectX-9, and Spectrum-6—form a synchronized architecture in which scale-up and scale-out fabrics are purpose-fit to the rack's GPU-CPU topology, constraining rack design to NVIDIA's reference architecture to achieve rated performance. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]

## Open Questions

- What is the latency profile of Spectrum-X co-packaged optics versus Quantum-X800 InfiniBand at scale? The evidence quantifies bandwidth and power efficiency but does not provide comparative latency figures for InfiniBand vs. Ethernet in AI training workloads.
- How does the NVLink security domain boundary interact with multi-tenant cloud deployments? The evidence confirms rack-scale Confidential Computing but does not address whether NVL72 racks can be logically partitioned for different tenants.
- What are the physical cabling and patching requirements for Quantum-X800 InfiniBand versus Spectrum-X Ethernet at the top-of-rack and spine layers? The evidence describes chip capabilities but not deployment cabling topology.
- The evidence (E014) describes a 'custom Spectrum-X-based interconnect' between LPX and NVL72 racks with medium confidence—what is the exact bandwidth, latency, and topology of this inter-rack link?
- Will Spectrum-7 and ConnectX-10 (referenced in the roadmap, E012) support backward compatibility with ConnectX-9 endpoints, or will full fabric upgrades be required when transitioning to the Rubin Ultra generation?
- How does NCCL leverage in-network compute within NVLink 6 switches specifically? The evidence confirms the capability exists but does not describe the programming model or which collective operations are offloaded.

## Source Notes

### Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf

- **Evidence ID:** E001
  **Claim:** NVLink 6 provides 3.6 TB/s of GPU-to-GPU bandwidth per GPU and 260 TB/s of rack-scale connectivity in the Vera Rubin NVL72, doubling the bandwidth of Blackwell's NVLink 5.
  **Evidence:** "The sixth-generation NVLink delivers a major leap for NVIDIA's high-speed GPU interconnect fabric that unifies 72 NVIDIA Rubin GPUs into a single performance domain. Doubling NVIDIA Blackwell's performance, the Rubin GPU delivers 3.6 terabytes per second (TB/s) of bandwidth per GPU and 260 TB/s of connectivity with low latency to facilitate faster communication."
  **Category:** networking
  **Relevance:** Directly quantifies NVLink 6's role in scale-up GPU-to-GPU communication within the AI data center rack.
  **Confidence:** high
- **Evidence ID:** E002
  **Claim:** NVLink fabric in the Vera Rubin NVL72 creates a unified, trusted execution environment spanning all 36 Vera CPUs, 72 Rubin GPUs, and the NVLink fabric for confidential computing at rack scale.
  **Evidence:** "The third generation of NVIDIA Confidential Computing expands security to full-rack scale with NVIDIA Vera Rubin NVL72. This platform creates a unified, trusted execution environment across all 36 NVIDIA Vera CPUs, 72 NVIDIA Rubin GPUs, and the NVIDIA NVLink™ fabric that seamlessly connects them. The platform maintains data security across CPU, GPU, and NVLink domains."
  **Category:** networking
  **Relevance:** Demonstrates NVLink's role not just as a performance interconnect but also as a security domain boundary in AI data centers.
  **Confidence:** high
- **Evidence ID:** E003
  **Claim:** The Vera Rubin NVL72 scales out across GPU clusters using NVIDIA Quantum-X800 InfiniBand and NVIDIA Spectrum-X Ethernet, complementing the NVLink scale-up fabric.
  **Evidence:** "NVIDIA Vera Rubin NVL72 unifies 72 NVIDIA Rubin GPUs, 36 NVIDIA Vera CPUs, NVIDIA ConnectX®-9 SuperNIC™ cards, and NVIDIA BlueField®-4 DPUs, and sits alongside NVIDIA LPX racks in a data center for fast, low-latency inference. It scales up intelligence in a rack-scale platform with the sixth-generation NVLink and NVLink Switch and scales out with NVIDIA Quantum-X800 InfiniBand and NVIDIA Spectrum-X™ Ethernet to power the AI industrial revolution at scale."
  **Category:** networking
  **Relevance:** Describes the complementary scale-out networking role of InfiniBand and Spectrum-X Ethernet alongside NVLink in the AI data center architecture.
  **Confidence:** high

### Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf

- **Evidence ID:** E004
  **Claim:** The Vera Rubin platform includes six co-designed chips—among them the ConnectX-9 SuperNIC and the Spectrum-6 Ethernet switch—to form a synchronized scale-out networking fabric.
  **Evidence:** "NVIDIA ConnectX-9: High-throughput, low-latency networking interface at the endpoint for scale-out AI. … NVIDIA Spectrum-6 Ethernet switch: Scale-out connectivity using co-packaged optics for efficiency and reliability. Together, these chips form a synchronized architecture in which GPUs execute transformer-era workloads, CPUs orchestrate data and control flow, scale-up and scale-out fabrics move tokens and state efficiently, and dedicated infrastructure processors operate and secure the AI factory itself."
  **Category:** networking
  **Relevance:** Identifies ConnectX-9 and Spectrum-6 as co-designed networking chips integral to the AI factory's scale-out architecture.
  **Confidence:** high

### Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf

- **Evidence ID:** E005
  **Claim:** NVLink 6 switch delivers 3.6 TB/s all-to-all scale-up bandwidth per GPU, enabling high-speed GPU-to-GPU communications for AI within the rack.
  **Evidence:** "NVIDIA NVLink 6 Switch: NVLink 6 switches feature 3.6 terabytes per second (TB/s) of all-to-all, scale-up bandwidth per GPU, enabling high-speed GPU-to-GPU communications for AI."
  **Category:** networking
  **Relevance:** Confirms NVLink 6's specific bandwidth figure and its role as the intra-rack GPU communications backbone.
  **Confidence:** high
- **Evidence ID:** E006
  **Claim:** ConnectX-9 SuperNICs deliver 1.6 Tb/s of per-GPU bandwidth with programmable RDMA for low-latency, GPU-direct networking at massive scale.
  **Evidence:** "NVIDIA ConnectX-9 SuperNIC: ConnectX‑9 SuperNICs deliver 1.6 terabits per second (Tb/s) of per-GPU bandwidth, with programmable remote direct-memory access (RDMA) for low‑latency, GPU‑direct networking at massive scale."
  **Category:** networking
  **Relevance:** Specifies ConnectX-9's bandwidth and RDMA capability as the scale-out NIC for AI data center GPU clusters.
  **Confidence:** high
- **Evidence ID:** E007
  **Claim:** Spectrum-X Ethernet co-packaged optics switches deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime over traditional networking with pluggable transceivers.
  **Evidence:** "NVIDIA Spectrum‑X Ethernet Co-Packaged Optics: Spectrum‑X Ethernet scale‑out switches with integrated silicon photonics deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime over traditional networking with pluggable transceivers."
  **Category:** networking
  **Relevance:** Quantifies Spectrum-X's advantages in power efficiency and uptime for AI data center Ethernet scale-out networking.
  **Confidence:** high

### NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf

- **Evidence ID:** E008
  **Claim:** The Rubin platform uses extreme co-design across six chips—including NVLink 6 Switch, ConnectX-9 SuperNIC, and Spectrum-6 Ethernet Switch—to slash training time and inference token generation cost.
  **Evidence:** "The Rubin platform uses extreme codesign across the six chips — the NVIDIA Vera CPU, NVIDIA Rubin GPU, NVIDIA NVLink™ 6 Switch, NVIDIA ConnectX®-9 SuperNIC, NVIDIA BlueField®-4 DPU and NVIDIA Spectrum™-6 Ethernet Switch — to slash training time and inference token costs."
  **Category:** networking
  **Relevance:** Confirms all three scale-out/scale-up networking chips are co-designed as part of the AI factory platform to reduce cost and latency.
  **Confidence:** high
- **Evidence ID:** E009
  **Claim:** NVLink 6 delivers 3.6 TB/s per GPU and 260 TB/s rack-wide bandwidth—described as more bandwidth than the entire internet—with in-network compute to accelerate collective operations.
  **Evidence:** "Sixth-Generation NVIDIA NVLink: Delivers the fast, seamless GPU-to-GPU communication required for today's massive MoE models. Each GPU offers 3.6TB/s of bandwidth, while the Vera Rubin NVL72 rack provides 260TB/s — more bandwidth than the entire internet. With built-in, in-network compute to speed collective operations, as well as new features for enhanced serviceability and resiliency, NVIDIA NVLink 6 switch enables faster, more efficient AI training and inference at scale."
  **Category:** networking
  **Relevance:** Highlights NVLink 6's critical role in supporting large MoE model training and inference through massive GPU-to-GPU bandwidth and in-network compute.
  **Confidence:** high

### 9d2c3468-1af8-4ed1-9b85-31bdebff03dc.pdf

- **Evidence ID:** E010
  **Claim:** Spectrum-X brings AI performance to Ethernet and, combined with silicon photonics, enables connection of hundreds of thousands of GPUs while saving megawatts of power.
  **Evidence:** "To build AI factories at scale, we had to reinvent networking. Spectrum-X brings AI performance to Ethernet, and with our silicon photonics technology, we can now connect hundreds of thousands of GPUs while saving megawatts of power. This is the network fabric of the AI era."
  **Category:** networking
  **Relevance:** Positions Spectrum-X as the Ethernet solution specifically optimized for AI data center scale, tied to silicon photonics power savings.
  **Confidence:** high

### Updates from NVIDIA GTC 2025 Conference.pdf

- **Evidence ID:** E011
  **Claim:** Co-packaged silicon photonics networking switches allow AI factories to scale to millions of GPUs, providing 3.5x power efficiency and 10x higher resiliency versus pluggable optics.
  **Evidence:** "Co-packaged silicon photonics networking switches to scale AI factories to millions of GPUs … 3.5X Power efficiency, 10X Higher resiliency, 1.3X Time to operation … 432 Transceivers Replaced [by] 72 NVIDIA Photonics."
  **Category:** networking
  **Relevance:** Quantifies the power and resiliency benefits of NVIDIA's co-packaged optics approach used in both Quantum-X (InfiniBand) and Spectrum-X (Ethernet) switches.
  **Confidence:** high
- **Evidence ID:** E012
  **Claim:** The NVIDIA AI platform roadmap shows a progression from Spectrum-5 through Spectrum-6 (with co-packaged optics) to Spectrum-7 alongside NVLink generation upgrades and ConnectX NIC generations.
  **Evidence:** "COMPUTE … Blackwell … Blackwell Ultra … Rubin … Rubin Ultra … NVLINK (SCALE-UP): 5th Gen NVL 72 1800 GB/s … 6th Gen NVSwitch 3600 GB/s … NETWORKING (SCALE-OUT): Spectrum5 51T, CX8 800G → Spectrum6 102T, CPO, CX9 1600G → Spectrum7 204T, CPO, CX10."
  **Category:** networking
  **Relevance:** Illustrates the generational roadmap of NVIDIA's scale-out (Spectrum/ConnectX) and scale-up (NVLink) networking technologies underpinning AI data centers.
  **Confidence:** high
- **Evidence ID:** E017
  **Claim:** The DGX B300 system is equipped with NVIDIA ConnectX-8 high-speed networking at 800 Gb/s for AI reasoning workloads.
  **Evidence:** "DGX B300 … Equipped with NVIDIA ConnectX-8 high speed networking at 800Gb/s."
  **Category:** networking
  **Relevance:** Confirms ConnectX-8 as the networking interface for the Blackwell Ultra DGX system, showing progression toward ConnectX-9 at 1.6 Tb/s in the Rubin generation.
  **Confidence:** high
- **Evidence ID:** E018
  **Claim:** The NVIDIA AI platform chips purpose-built for AI supercomputing span GPU, CPU, DPU, NIC, NVLink Switch, IB Switch, and Ethernet Switch, all unified by cluster-scale software including NCCL.
  **Evidence:** "Chips Purpose-Built for AI Supercomputing: GPU | CPU | DPU | NIC | NVLink Switch | IB Switch | Enet Switch … CUDA • DOCA • NCCL [Cluster-Scale Software]."
  **Category:** architecture
  **Relevance:** Illustrates the full breadth of co-designed networking chips—NVLink Switch for scale-up, InfiniBand Switch and Ethernet Switch for scale-out—that collectively form the AI data center network fabric.
  **Confidence:** high

### NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf

- **Evidence ID:** E013
  **Claim:** The Vera Rubin NVL72 scales out with Quantum-X800 InfiniBand and Spectrum-X Ethernet to sustain high GPU cluster utilization and reduce time-to-train and total cost of ownership.
  **Evidence:** "The NVL72 scales seamlessly with NVIDIA Quantum-X800 InfiniBand and Spectrum-X Ethernet to sustain high utilization across massive GPU clusters while reducing time-to-train and total cost of ownership."
  **Category:** networking
  **Relevance:** Confirms InfiniBand and Spectrum-X Ethernet as the scale-out networking layers that sustain performance across large GPU clusters in AI data centers.
  **Confidence:** high
- **Evidence ID:** E014
  **Claim:** When LPX and Vera Rubin NVL72 are co-deployed, they are connected via a custom Spectrum X-based interconnect, enabling cooperative decode token processing across the two rack types.
  **Evidence:** "When co-deployed with the VR NVL72 via a custom Spectrum X-based interconnect, the two racks are supposed to process every decode token cooperatively with attention decode existing on Rubin and Feed Forward layers being offloaded to LPUs."
  **Category:** networking
  **Relevance:** Shows Spectrum-X being used not only for broad cluster scale-out but also as the inter-rack fabric connecting Rubin and LPX racks for disaggregated inference.
  **Confidence:** medium

### NVIDIA-based Enterprise Solutions.pdf

- **Evidence ID:** E015
  **Claim:** The GB300 NVL72 is built for AI reasoning and based on Quantum-X800 InfiniBand or Spectrum-X Ethernet paired with ConnectX-8 SuperNIC for scale-out networking.
  **Evidence:** "A fully liquid-cooled, rack-scale design optimized for test-time scaling inference, delivering up to 50× higher output for reasoning model inference compared to the NVIDIA Hopper™ platform, based on NVIDIA Quantum-X800 InfiniBand or Spectrum™-X Ethernet paired with ConnectX®-8 SuperNIC™."
  **Category:** networking
  **Relevance:** Documents InfiniBand and Spectrum-X Ethernet as alternative scale-out networking options for the Blackwell Ultra generation rack system.
  **Confidence:** high
- **Evidence ID:** E016
  **Claim:** The GB200 NVL72 compute tray includes ConnectX-7 NICs and BlueField-3 DPUs for networking, while using NVLink Switch trays with 1.8 TB/s GPU-GPU interconnect for scale-up.
  **Evidence:** "4 x 400Gb/s OSFP (NVIDIA ConnectX®-7 NIC), 2 x NVIDIA® BlueField®-3 DPUs … NVIDIA NVLink™ Switch Trays - 144 x NVLink™ ports per tray - Fifth-generation NVLink™ with 1.8TB/s GPU-GPU interconnect."
  **Category:** networking
  **Relevance:** Shows the evolution of ConnectX NICs and NVLink generations across Blackwell-generation AI data center racks, providing a comparison baseline for newer generations.
  **Confidence:** high

## Evaluation Warnings

- None.
