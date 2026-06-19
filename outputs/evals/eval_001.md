# Research Memo: How do Rubin-class racks affect power distribution, UPS/BBU strategy, and utility interconnect?

**Question:** How do Rubin-class racks affect power distribution, UPS/BBU strategy, and utility interconnect?

## Executive Summary

Rubin-class (NVL72) racks represent a fundamental shift in data center power architecture. With extreme compute density (3.6 ExaFLOPS per rack), fully liquid-cooled rack-scale design, and NVIDIA's explicit targeting of gigawatt-scale AI factories, Rubin-class deployments demand a wholesale rethinking of power distribution, UPS/BBU strategy, and utility interconnect planning. Power delivery is a first-class, co-designed element of the platform—not an afterthought. At the facility level, the "data center as the unit of compute" philosophy mandates holistic, factory-scale power and resilience planning rather than per-rack or per-server approaches. Networking efficiency gains from co-packaged optics partially offset raw compute power demands, and the platform's proactive RAS Engine reduces unplanned fault events—together moderating (but not eliminating) the UPS/BBU burden. Utility interconnects must be engineered for gigawatt-class delivery as Rubin Ultra (NVL576) configurations follow.

## Confirmed Facts

- Rubin-class NVL72 racks are fully liquid-cooled rack-scale systems with power delivery co-designed alongside compute, networking, and cooling as a single integrated platform. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E001]
- The Vera Rubin NVL72 rack is built on the third-generation MGX NVL72 rack design, featuring cable-free modular tray designs and support from over 80 MGX ecosystem partners for rapid deployment. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E002]
- The NVL72 rack family (as seen in the GB300 Blackwell Ultra predecessor) uses three 1U 33kW power shelves on each side of the rack, compatible with in-rack or in-row CDU cooling. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- The Rubin platform's modular, cable-free tray design enables up to 18x faster assembly and servicing than Blackwell, reducing maintenance overhead including for power-related components. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]
- The second-generation RAS Engine provides real-time health checks, fault tolerance, and proactive maintenance across GPU, CPU, and NVLink to maximize system productivity and continuous operation. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E005]
- Vera Rubin NVL72 delivers 3,600 PFLOPS NVFP4 inference and 2,520 PFLOPS NVFP4 training from a single rack, representing extreme power density per rack unit. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E006]
- The Vera Rubin platform treats the data center—not a single GPU server—as the unit of compute, requiring power planning and utility interconnect sizing to be conceived at factory scale. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]
- Spectrum-X co-packaged optics switches deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime versus traditional pluggable-transceiver networking. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E008]
- NVIDIA's roadmap explicitly targets gigawatt-scale AI factories, with Rubin NVL72 liquid-cooled systems in 2026 followed by Rubin Ultra NVL576 liquid-cooled configurations. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E009]
- Co-packaged optics in Quantum-X and Spectrum-X switches replace 432 pluggable transceivers with 72 NVIDIA Photonics units, achieving 3.5x power efficiency improvement and enabling approximately 3x more GPUs at ISO Power budgets. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E010]
- The LPX rack (256 Groq 3 LPUs) is fully liquid-cooled, built on MGX infrastructure, and will be co-deployed alongside Vera Rubin NVL72 racks in the second half of 2026. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]
- The Vera Rubin NVL72 platform delivers up to 10x more inference throughput per watt compared to Blackwell across the full spectrum of AI workloads. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E012]

## Inferences

- The NVL72 power shelf architecture (3x 33kW shelves per side, ~198kW total capacity per rack) confirmed for the GB300 Blackwell Ultra generation likely carries forward to Rubin NVL72 given the shared third-generation MGX NVL72 rack form factor, but this has not been explicitly confirmed for Rubin-class hardware specifically.
- Gigawatt-scale AI factory power targets imply that utility interconnects for Rubin deployments will require dedicated high-voltage substation infrastructure and potentially new transmission agreements with grid operators—far beyond the scope of traditional enterprise UPS/generator topologies.
- The 10x throughput-per-watt improvement means that operators consolidating from Blackwell to Rubin may be able to achieve the same inference SLA with significantly fewer racks and proportionally lower UPS/BBU capacity, even if per-rack power density remains high or increases.
- The proactive RAS Engine (E005) may reduce the frequency and duration of unplanned power interruption events, potentially allowing UPS/BBU systems to be right-sized for planned maintenance bridge windows rather than worst-case fault scenarios—though this requires validation against actual fault rate data.
- Co-deployment of LPX racks (E011) alongside NVL72 introduces a heterogeneous power load profile on the factory floor; UPS and PDU design must accommodate two distinct rack power signatures, potentially complicating load balancing and circuit sizing.
- Cable-free modular tray design (E004) likely simplifies power bus routing within the rack and reduces the risk of power-cabling errors during rack assembly or hot-swap servicing, improving operational reliability at scale.

## Power Implications

- Rubin NVL72 racks consume extreme per-rack power density, driven by 3,600 PFLOPS NVFP4 inference and 2,520 PFLOPS NVFP4 training compute from 72 GPUs and 36 CPUs in a single rack enclosure. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E006]
- The NVL72 rack family uses three 1U 33kW power shelves on each side (~198kW total shelf capacity per rack), establishing the in-rack power distribution architecture for this form factor. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- Power delivery is a first-class co-designed element of the Rubin platform, architected together with GPUs, CPUs, networking, and cooling as a single system rather than optimized in isolation. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E001]
- NVIDIA's roadmap explicitly targets gigawatt-scale AI factories with Rubin and Rubin Ultra NVL576 configurations, requiring utility interconnects to be engineered for gigawatt-class power delivery at the facility level. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E009]
- Co-packaged optics (Quantum-X/Spectrum-X) replace 432 pluggable transceivers with 72 NVIDIA Photonics units, achieving 3.5x networking power efficiency and enabling approximately 3x more GPUs to be deployed at a fixed ISO Power budget. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E010]
- Spectrum-X networking switches deliver 5x better power efficiency versus traditional pluggable-transceiver networking, partially offsetting compute power draw in total facility power budgets. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E008]
- The Vera Rubin NVL72 delivers 10x more inference throughput per watt versus Blackwell, meaning UPS capacity needed per unit of inference output is substantially reduced even as absolute per-rack power density remains extreme. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E012]
- Co-deployment of LPX racks (256 Groq 3 LPUs, fully liquid-cooled) alongside NVL72 racks increases the aggregate facility power envelope and introduces a second distinct power load profile that must be accounted for in UPS and utility interconnect planning. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]
- The 'data center as the unit of compute' design philosophy mandates that utility interconnect capacity, UPS topology, and power distribution be planned holistically at AI factory scale rather than on a per-rack basis. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]

## Cooling Implications

- Rubin NVL72 racks are fully liquid-cooled rack-scale systems, with cooling co-designed alongside compute and power delivery as part of the integrated platform architecture. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E001]
- The NVL72 power shelf architecture is compatible with both in-rack CDU and in-row CDU cooling configurations, providing flexibility in facility cooling topology design. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- The LPX rack is fully liquid-cooled and built on MGX infrastructure, meaning the AI factory floor will require liquid cooling infrastructure capable of serving both NVL72 and LPX rack types simultaneously. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]
- The cable-free modular tray design enables 18x faster assembly and serviceability versus Blackwell, reducing the complexity and duration of maintenance events that could interrupt cooling circuits within the rack. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]

## Networking Implications

- Spectrum-X Ethernet scale-out switches with integrated silicon photonics deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime over traditional pluggable-transceiver networking—reducing total AI factory networking power load and improving uptime. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E008]
- Co-packaged optics replace 432 pluggable transceivers with 72 NVIDIA Photonics units across Quantum-X and Spectrum-X switches, achieving 3.5x power efficiency improvement and 10x higher resiliency at the networking layer. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E010]
- The Rubin platform integrates software-defined NVLink routing within the rack, enabling intelligent resiliency and continuous operation that reduces networking-related downtime events and their impact on UPS bridge requirements. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]

## Rack Architecture Implications

- The Vera Rubin NVL72 is built on the third-generation MGX NVL72 rack design, offering a seamless generational transition with support from over 80 MGX ecosystem partners for deployment and supply chain continuity. [Source: Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf, Evidence: E002]
- The cable-free modular tray design enables 18x faster rack assembly and servicing versus Blackwell, directly reducing power-cabling complexity and the risk of errors during deployment or hot-swap maintenance. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E004]
- The NVL72 rack houses three 1U 33kW power shelves on each side of the rack, establishing a symmetric, high-density in-rack power distribution structure at approximately 198kW total shelf capacity. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- The second-generation RAS Engine provides real-time health checks and proactive fault tolerance spanning GPU, CPU, and NVLink, supporting continuous rack operation and reducing unplanned downtime that would otherwise trigger UPS/BBU bridge events. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E005]
- The Rubin platform treats the data center as the unit of compute, meaning individual rack architecture decisions (power shelving, cooling hookup, networking) must be evaluated in the context of factory-scale deployment, not standalone rack optimization. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]
- Co-deployment of LPX racks (MGX-based, liquid-cooled) alongside NVL72 on the same factory floor introduces a heterogeneous rack architecture environment requiring unified power, cooling, and interconnect planning across two distinct rack form factors. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]

## Open Questions

- What is the confirmed total power draw (kW) of a fully loaded Rubin NVL72 rack? The 3x 33kW shelf architecture (~198kW) is confirmed for GB300 Blackwell Ultra but not yet explicitly published for Rubin-class hardware.
- How does NVIDIA define the UPS/BBU strategy for gigawatt-scale AI factories—does the platform assume utility-grade redundancy (e.g., dual utility feeds, on-site generation) rather than traditional rack-level UPS?
- What is the specific RAS Engine fault rate improvement versus Blackwell, and how does this translate to quantifiable reductions in UPS/BBU activation frequency or bridge duration requirements?
- Are Rubin Ultra NVL576 racks expected to use the same in-rack power shelf architecture as NVL72, or does the 576-GPU configuration require a fundamentally different power distribution approach?
- How are UPS/BBU systems expected to be architected at gigawatt factory scale—centralized string UPS, distributed rack-level BBU, or a hybrid topology—and does NVIDIA provide reference designs for this?
- What utility interconnect voltage levels (e.g., 480V, medium-voltage direct) are recommended or required for Rubin-class factory deployments, and are there published grid interconnect specifications from NVIDIA or its ODM partners?
- How does the co-deployment of LPX racks affect load-balancing strategies across shared UPS systems, and are the two rack types expected to be on separate power distribution branches or unified circuits?

## Source Notes

### Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf

- **Evidence ID:** E001
  **Claim:** Rubin-class racks (NVL72) are fully liquid-cooled rack-scale systems, with power delivery co-designed alongside compute, networking, and cooling as part of the platform.
  **Evidence:** "The NVIDIA Vera Rubin platform was designed specifically for this new reality. Extreme co-design is the foundation of the Vera Rubin platform. GPUs, CPUs, networking, security, software, power delivery, and cooling are architected together as a single system rather than optimized in isolation."
  **Category:** power
  **Relevance:** Establishes that power delivery is a first-class, co-designed element of the Vera Rubin rack architecture, not an afterthought, setting the context for how power distribution and cooling are treated at rack scale.
  **Confidence:** high
- **Evidence ID:** E004
  **Claim:** The Rubin platform's modular, cable-free tray design enables up to 18x faster assembly and servicing than Blackwell, reducing maintenance overhead including for power-related components.
  **Evidence:** "The rack introduces modular, cable-free tray designs for 18x faster assembly and serviceability versus NVIDIA Blackwell, combined with intelligent resiliency and software-defined NVLink routing, which ensures continuous operation and reduces maintenance overhead."
  **Category:** operations
  **Relevance:** The cable-free modular tray design directly reduces the complexity of power cabling within the rack, affecting how power is distributed and how quickly power-related servicing can occur—relevant to UPS/BBU maintenance windows.
  **Confidence:** high
- **Evidence ID:** E007
  **Claim:** The Vera Rubin platform treats the data center as the unit of compute, meaning power planning, utility interconnect sizing, and UPS strategy must be conceived at factory scale rather than per-server.
  **Evidence:** "By doing so, the Vera Rubin platform treats the data center, not a single GPU server, as the unit of compute. This approach establishes a new foundation for producing intelligence efficiently, securely, and predictably at scale. It ensures that performance and efficiency hold up in production deployments, not just isolated component benchmarks."
  **Category:** architecture
  **Relevance:** Framing the data center as the unit of compute means utility interconnect capacity, UPS topology, and power distribution must be planned holistically at AI factory scale, not rack by rack.
  **Confidence:** high

### Rack-Scale Agentic AI Supercomputer _ NVIDIA Vera Rubin NVL72.pdf

- **Evidence ID:** E002
  **Claim:** The Vera Rubin NVL72 rack is built on the third-generation MGX NVL72 rack design, offering a seamless transition from prior generations with cable-free modular tray designs for 18x faster assembly and serviceability versus Blackwell.
  **Evidence:** "Vera Rubin NVL72 is built on the third-generation NVIDIA MGX™ NVL72 rack design, offering a seamless transition from prior generations. It delivers AI training with one-fourth the GPUs and AI inference at one-tenth the cost per million tokens versus NVIDIA Blackwell. Featuring cable‑free modular tray designs and support from over 80 MGX ecosystem partners, the rack-scale AI supercomputer delivers world‑class performance with rapid deployment."
  **Category:** rack architecture
  **Relevance:** Describes the physical rack architecture of Rubin-class systems, including the modular, cable-free design that affects how power infrastructure is assembled and maintained within the rack.
  **Confidence:** high
- **Evidence ID:** E008
  **Claim:** The Spectrum-X co-packaged optics switches deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptime versus traditional pluggable-transceiver networking, reducing total AI factory power load at the networking layer.
  **Evidence:** "Spectrum‑X Ethernet scale‑out switches with integrated silicon photonics deliver 5x better power efficiency, 10x higher network resiliency, and up to 5x more uptim e over traditional networking with pluggable transceivers."
  **Category:** power
  **Relevance:** Networking power efficiency directly affects overall rack and facility power budgets. Lower networking power draw partially offsets the high compute power draw, influencing how UPS capacity is sized and how utility interconnect headroom is allocated.
  **Confidence:** high

### NVIDIA-based Enterprise Solutions.pdf

- **Evidence ID:** E003
  **Claim:** The GB300 NVL72 (Blackwell Ultra predecessor to Rubin) uses three 1U 33kW power shelves on each side of the rack, compatible with in-rack or in-row CDU cooling—indicating a similar high-density power shelf architecture applies to the NVL72 rack family.
  **Evidence:** "3 x 1U 33kW Power Shelves [listed twice, on each side] Compatible with in-rack CDU or in-row CDU"
  **Category:** power
  **Relevance:** Provides concrete power shelf specifications (3x 33kW shelves per side = up to ~198kW total capacity) for the NVL72 rack form factor shared across Blackwell and Rubin generations, directly relevant to how power distribution is structured within a Rubin-class rack.
  **Confidence:** medium
- **Evidence ID:** E006
  **Claim:** Vera Rubin NVL72 delivers 3.6 ExaFLOPS NVFP4 inference and 2.5 ExaFLOPS training from a single rack, representing an extreme power density that directly impacts utility interconnect requirements.
  **Evidence:** "NVIDIA Vera Rubin NVL72 unifies leading-edge technologies from NVIDIA: 72 Rubin GPUs, 36 Vera CPUs... 3,600 PFLOPS [NVFP4 Inference] 2,520 PFLOPS [NVFP4 Training]"
  **Category:** power
  **Relevance:** The extreme compute density of the Rubin NVL72 rack implies proportionally extreme power draw per rack, which is central to understanding how utility interconnects must be scaled and how UPS/BBU must be sized.
  **Confidence:** high

### NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf

- **Evidence ID:** E005
  **Claim:** The second-generation RAS Engine in the Rubin platform provides real-time health checks and proactive maintenance to maximize system productivity, supporting continuous operation without downtime—reducing reliance on reactive BBU/UPS interventions.
  **Evidence:** "Second-Generation RAS Engine: The Rubin platform — spanning GPU, CPU and NVLink — features real-time health checks, fault tolerance and proactive maintenance to maximize system productivity. The rack's modular, cable-free tray design enables up to 18x faster assembly and servicing than Blackwell."
  **Category:** operations
  **Relevance:** The RAS Engine's proactive fault tolerance and real-time health monitoring capabilities reduce unplanned downtime, which affects how UPS and BBU systems need to be sized and triggered—a more resilient platform may reduce the frequency and duration of BBU bridge events.
  **Confidence:** medium

### Updates from NVIDIA GTC 2025 Conference.pdf

- **Evidence ID:** E009
  **Claim:** NVIDIA's roadmap projects Rubin-class NVL72 systems (liquid-cooled) in 2026, followed by Rubin Ultra in NVL576 liquid-cooled configurations, pointing toward gigawatt-scale AI factory power requirements at the facility level.
  **Evidence:** "Rubin 8S HBM4 ... Oberon NVL72 Liquid Cooled ... Rubin Ultra 16S HBM4e ... Kyber NVL576 Liquid Cooled ... NVIDIA Paves Road to Gigawatt AI Factories One-Year Rhythm | Full-Stack | One Architecture | CUDA Everywhere"
  **Category:** power
  **Relevance:** The explicit reference to 'Gigawatt AI Factories' alongside the Rubin roadmap signals that utility interconnects must be designed for gigawatt-class power delivery, fundamentally reshaping UPS strategy and grid interconnect design for facilities deploying Rubin-class racks.
  **Confidence:** high
- **Evidence ID:** E010
  **Claim:** Co-packaged optics in Quantum-X and Spectrum-X switches replace 432 pluggable transceivers with 72 NVIDIA Photonics units, achieving 3.5x power efficiency improvement—directly reducing the networking portion of total facility power draw.
  **Evidence:** "NVIDIA Photonics Solves Power and Reliability Challenges of AI Scale-Out ... 432 Transceivers Replaced [by] 72 NVIDIA Photonics ... 3.5X Power efficiency ... 10X Higher resiliency ... ~3X the GPUs at ISO Power"
  **Category:** power
  **Relevance:** At fixed utility power budgets (ISO Power), co-packaged optics allow approximately 3x more GPUs to be deployed, meaning utility interconnects and UPS systems can support significantly more Rubin-class compute without additional capacity upgrades.
  **Confidence:** high

### NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf

- **Evidence ID:** E011
  **Claim:** The LPX rack (housing 256 Groq 3 LPUs) is fully liquid-cooled and built on MGX infrastructure, co-deployed alongside Vera Rubin NVL72 racks—adding a second distinct power load profile to the AI factory floor that must be accounted for in UPS and utility interconnect planning.
  **Evidence:** "The LPX Rack is fully liquid-cooled, built on MGX infrastructure, and will be available in the second half of 2026, coincident with the broader Vera Rubin rollout."
  **Category:** power
  **Relevance:** When LPX racks are co-deployed with Vera Rubin NVL72 racks, the combined power envelope of the AI factory increases. UPS and utility interconnect planning must accommodate the aggregate power draw of both rack types.
  **Confidence:** medium
- **Evidence ID:** E012
  **Claim:** The Vera Rubin NVL72 platform delivers 10x more inference throughput per watt compared to Blackwell, improving power use efficiency and potentially reducing the UPS capacity needed per unit of inference output.
  **Evidence:** "NVIDIA claims the platform delivers up to 10x more inference throughput per watt and one-tenth the cost per token compared to the Blackwell generation across the full spectrum of AI workloads."
  **Category:** power
  **Relevance:** Higher throughput per watt means operators can achieve the same inference output with less total power draw, which influences how UPS systems are sized relative to workload SLA requirements and how utility interconnect capacity is contracted.
  **Confidence:** high

## Evaluation Warnings

- None.
