# Research Memo: What cooling infrastructure changes are implied by liquid-cooled AI factory racks?

**Question:** What cooling infrastructure changes are implied by liquid-cooled AI factory racks?

## Executive Summary

Liquid-cooled AI factory racks—spanning NVIDIA's Blackwell, Vera Rubin, and roadmap Kyber generations—represent a definitive, permanent departure from traditional air-cooled data center infrastructure. Every confirmed rack design (GB200 NVL72, GB300 NVL72, DGX Rubin NVL8, LPX Rack, and Grace CPU DLC nodes) mandates active liquid cooling via cold plates and coolant distribution units (CDUs). Power densities approaching ~99 kW per rack make air cooling physically insufficient. Cooling is now a first-class co-design element alongside compute, networking, and power delivery, meaning facilities must plan, provision, and scale liquid cooling infrastructure in lockstep with AI compute deployments. OEM partners (e.g., Lenovo Neptune) are already building proprietary liquid-cooling solutions, confirming that the broader supply chain is adapting to this new baseline requirement.

## Confirmed Facts

- The GB200 NVL72 is a fully liquid-cooled rack-scale system that interconnects all nodes with NVLink technology. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- The GB300 NVL72 is a fully liquid-cooled, rack-scale design optimized for test-time scaling inference workloads. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- Liquid-cooled GB200/GB300 NVL72 compute trays use Superchip cold plate loops (2 per 1U node) as the primary thermal management mechanism. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E004]
- The GB300 NVL72 rack incorporates three 33 kW (1U) power shelves (~99 kW total) and requires compatibility with in-rack or in-row CDUs. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E005]
- Liquid-cooled rack-scale AI systems require data centers to support compatible in-rack or in-row cooling distribution units (CDUs). [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E001]
- The NVIDIA DGX Rubin NVL8 is a liquid-cooled AI system powered by eight Rubin GPUs, confirming liquid cooling is required even for smaller Rubin-generation configurations. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E006]
- The Vera Rubin platform was designed with extreme co-design across compute, networking, power delivery, cooling, and system architecture to enable AI factory scale operation. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]
- AI factory racks must operate within tightly constrained power and cooling limits, making cooling infrastructure capacity a hard constraint on scale-out. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E008]
- NVIDIA's multi-year roadmap explicitly labels both the Rubin NVL72 and the next-generation Kyber NVL576 as liquid-cooled, confirming liquid cooling as the long-term infrastructure standard. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E009]
- The LPX Rack, housing 256 Groq 3 LPUs for decode acceleration, is fully liquid-cooled and built on MGX infrastructure. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]
- The Grace CPU Superchip server offers a direct liquid cooling (DLC) variant with leak detection, indicating CPU-only nodes are also transitioning to liquid cooling. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E013]
- Lenovo is integrating its Neptune liquid-cooling solution with NVIDIA Rubin platform racks for enterprise AI factory deployments. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E014]

## Inferences

- Because every rack type in an AI factory—GPU compute (NVL72), inference accelerator (LPX), and CPU (Grace DLC)—requires liquid cooling, a facility operator cannot selectively deploy liquid cooling for only GPU racks; the entire pod or row must be liquid-cooling-capable.
- The shift to modular, cable-free tray designs in the Vera Rubin NVL72 likely simplifies standardized coolant manifold connections, reducing the risk of leaks and maintenance downtime during hot-swap operations—though this is inferred from mechanical design intent rather than explicit coolant plumbing documentation.
- The ~99 kW per rack power density of the GB300 NVL72 effectively eliminates air cooling as a viable option, meaning any data center still relying solely on CRAC/CRAH air cooling cannot support these racks without a facility retrofit.
- Power efficiency gains from co-packaged silicon photonics networking (3.5× power efficiency improvement) may partially offset aggregate thermal load at the facility level, indirectly easing peak CDU capacity requirements—but the magnitude of this offset relative to compute heat loads is not quantified in the evidence.
- OEM participation (e.g., Lenovo Neptune) signals that a multi-vendor liquid-cooling ecosystem is forming, which should reduce procurement risk and increase solution availability over time.
- The permanent roadmap commitment to liquid cooling (Rubin NVL72 through Kyber NVL576) implies that data center operators who invest in liquid cooling infrastructure now will not face obsolescence risk across at least two hardware generations.

## Power Implications

- The GB300 NVL72 rack incorporates three 33 kW power shelves, yielding approximately 99 kW of rack-level power, which is a primary driver of mandatory liquid cooling adoption. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E005]
- AI factory racks must operate within tightly constrained power limits, making power provisioning a hard co-constraint alongside cooling capacity during facility design. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E008]
- Co-packaged silicon photonics networking switches deliver approximately 3.5× power efficiency gains, which may reduce aggregate facility power and heat load contributions from the networking layer. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E012]

## Cooling Implications

- Data centers deploying NVL72-class racks must install compatible in-rack CDUs or in-row CDUs to supply and return coolant to Superchip cold plate loops in each compute tray. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E001]
- The full liquid-cooled architecture of the GB200 NVL72 requires facilities to transition away from traditional air-cooled CRAC/CRAH infrastructure toward active liquid cooling distribution systems. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E002]
- Liquid cooling is consistently required across the entire Blackwell NVL72 rack generation (both GB200 and GB300 variants), confirming it is not a premium option but a mandatory infrastructure baseline. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E003]
- Superchip cold plate loops replace traditional air-cooled heat sinks at the compute tray level, requiring coolant plumbing (supply/return manifolds) to be integrated into the rack structure. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E004]
- Liquid cooling requirements extend to smaller AI system form factors: the DGX Rubin NVL8 (8-GPU system) also mandates liquid cooling, meaning even non-hyperscale deployments require CDU infrastructure. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E006]
- Cooling is a first-class co-design element of the Vera Rubin platform, meaning facility cooling systems must be planned and scaled in concert with compute and networking infrastructure rather than retrofitted after the fact. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]
- NVIDIA's roadmap through at least the Kyber NVL576 generation confirms liquid cooling is the permanent, long-term infrastructure standard for AI factory racks—not a transitional measure. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E009]
- Liquid cooling requirements extend beyond GPU compute racks to specialized inference accelerator racks (LPX), meaning all rack types in an AI factory deployment require liquid cooling infrastructure. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]
- Direct liquid cooling with leak detection is being applied to CPU server nodes (Grace Superchip DLC), confirming a broad infrastructure shift toward liquid cooling across all rack components, not only GPU nodes. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E013]
- OEM partners such as Lenovo are developing proprietary liquid-cooling solutions (Neptune) specifically for AI factory rack deployments, indicating the supply chain is actively adapting to liquid cooling as the new baseline. [Source: NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf, Evidence: E014]

## Networking Implications

- Co-packaged silicon photonics networking switches are designed to scale AI factories to millions of GPUs at approximately 3× the GPU density at iso-power and 3.5× power efficiency, indirectly reducing thermal load contributions from the networking layer. [Source: Updates from NVIDIA GTC 2025 Conference.pdf, Evidence: E012]
- The Vera Rubin platform applies extreme co-design across compute, networking, power delivery, and cooling, meaning networking infrastructure upgrades must be planned in concert with cooling infrastructure upgrades to avoid bottlenecks. [Source: Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf, Evidence: E007]

## Rack Architecture Implications

- The GB200/GB300 NVL72 rack-scale design integrates cold plate loops at the 1U compute tray level, requiring racks to include internal coolant manifolds and CDU-compatible quick-connect fittings as structural components. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E004]
- The GB300 NVL72 rack incorporates three dedicated 1U 33 kW power shelves alongside compute trays, indicating rack real estate must be allocated for high-density power delivery in addition to compute and cooling components. [Source: NVIDIA-based Enterprise Solutions.pdf, Evidence: E005]
- The Vera Rubin NVL72 rack introduces modular, cable-free tray designs enabling 18× faster assembly and serviceability compared to Blackwell, which also implies standardized liquid cooling loop connections at the tray level to support rapid hot-swap maintenance. [Source: Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf, Evidence: E010]
- The LPX Rack for Groq 3 LPU inference acceleration is built on MGX infrastructure and is fully liquid-cooled, confirming that MGX is the common mechanical and cooling framework spanning both GPU and non-GPU AI factory rack types. [Source: NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf, Evidence: E011]

## Open Questions

- What is the minimum facility water temperature (supply/return delta-T) required by CDUs supporting NVL72-class racks, and do existing chilled-water plants meet this specification without modification?
- How are coolant leak detection and emergency shutoff systems standardized across multi-vendor AI factory deployments (e.g., Lenovo Neptune vs. other OEM CDU solutions)?
- What is the net facility-level thermal load reduction, if any, from co-packaged silicon photonics networking relative to the increased compute heat load of NVL72/NVL576 racks?
- Do existing raised-floor or overhead plenum air-cooling systems need to be fully decommissioned, or can they serve as a supplemental/backup cooling path for non-GPU rack components?
- What coolant type (water, dielectric fluid, refrigerant) is specified for the Superchip cold plate loops, and what are the associated materials compatibility and maintenance requirements?
- As rack power densities increase from ~99 kW (GB300 NVL72) toward Kyber NVL576-class systems, what CDU capacity and facility chilled-water loop upgrades will be required, and on what timeline?
- How does the modular, cable-free tray design of the Vera Rubin NVL72 specifically implement coolant quick-connects, and are these standardized across OEM implementations?

## Source Notes

### NVIDIA-based Enterprise Solutions.pdf

- **Evidence ID:** E001
  **Claim:** Liquid-cooled rack-scale AI systems require data centers to support compatible in-rack or in-row cooling distribution units (CDUs).
  **Evidence:** "Compatible with in-rack CDU or in-row CDU ... 2 x Superchip cold plate loops ... 8 x 40x40x56mm fans"
  **Category:** cooling
  **Relevance:** Directly specifies that the liquid-cooled GB200 NVL72 compute trays use cold plate loops and require compatibility with either in-rack or in-row coolant distribution units, implying data center cooling infrastructure must be upgraded to support active liquid cooling.
  **Confidence:** high
- **Evidence ID:** E002
  **Claim:** The GB200 NVL72 rack-scale system is fully liquid-cooled, representing a departure from traditional air-cooled infrastructure.
  **Evidence:** "NVIDIA GB200 NVL72 ... A fully liquid-cooled rack-scale design that interconnects all nodes with NVIDIA NVLink™ technology and delivers the performance of 'one big GPU,' surpassing previous-generation GPU platforms with exceptional interconnect bandwidth and energy efficiency for AI and HPC workloads."
  **Category:** cooling
  **Relevance:** Confirms the GB200 NVL72 is fully liquid-cooled at rack scale, implying facilities must transition from air cooling to liquid cooling infrastructure.
  **Confidence:** high
- **Evidence ID:** E003
  **Claim:** The GB300 NVL72 rack is also fully liquid-cooled and optimized for AI reasoning inference workloads.
  **Evidence:** "NVIDIA GB300 NVL72 ... A fully liquid-cooled, rack-scale design optimized for test-time scaling inference, delivering up to 50× higher output for reasoning model inference compared to the NVIDIA Hopper™ platform."
  **Category:** cooling
  **Relevance:** Confirms liquid cooling is a consistent requirement across the Blackwell Ultra NVL72 rack generation, reinforcing that AI factory deployments depend on liquid-cooled infrastructure.
  **Confidence:** high
- **Evidence ID:** E004
  **Claim:** The liquid-cooled GB200 NVL72 compute trays use superchip cold plate loops as the primary thermal management mechanism.
  **Evidence:** "System Cooling: 1U Liquid-cooled server node ... 8 x 40x40x56mm fans, 2 x Superchip cold plate loops"
  **Category:** cooling
  **Relevance:** Identifies the specific liquid-cooling components (cold plate loops) used at the compute tray level, indicating that traditional air-cooled heat sinks are replaced by direct-liquid cold plates requiring coolant plumbing.
  **Confidence:** high
- **Evidence ID:** E005
  **Claim:** The NVIDIA GB300 NVL72 rack uses three 33kW power shelves and requires in-rack or in-row CDU support, highlighting the high power density that drives liquid cooling requirements.
  **Evidence:** "3 x 1U 33kW Power Shelves ... Compatible with in-rack CDU or in-row CDU"
  **Category:** cooling
  **Relevance:** Shows that each rack incorporates up to ~99kW of power shelving, which is a key driver for mandatory liquid cooling; air cooling would be insufficient at this power density.
  **Confidence:** high
- **Evidence ID:** E013
  **Claim:** The Grace CPU Superchip server offers a direct liquid cooling (DLC) variant with leak detection, indicating that even CPU-only nodes in AI factory racks are transitioning to liquid cooling.
  **Evidence:** "2U 4-node rear access DLC server ... Direct liquid cooling with leak detection"
  **Category:** cooling
  **Relevance:** Shows that direct liquid cooling with leak detection is being applied to CPU server nodes as well as GPU nodes, confirming a broad infrastructure shift toward liquid cooling across all rack components in AI factory deployments.
  **Confidence:** high

### Infrastructure for Scalable AI Reasoning _ NVIDIA Vera Rubin Platform.pdf

- **Evidence ID:** E006
  **Claim:** The NVIDIA DGX Rubin NVL8 AI system is liquid-cooled, indicating liquid cooling is required even for smaller Rubin-generation configurations.
  **Evidence:** "NVIDIA DGX Rubin NVL8 is a liquid-cooled AI system powered by eight NVIDIA Rubin GPUs and sixth-generation NVLink. It's purpose-built to accelerate training, inference, and post-training for every AI workload."
  **Category:** cooling
  **Relevance:** Shows that liquid cooling extends beyond the full NVL72 rack form factor down to smaller 8-GPU DGX systems, meaning liquid cooling infrastructure is a baseline requirement across the Rubin product family.
  **Confidence:** high
- **Evidence ID:** E010
  **Claim:** The Vera Rubin NVL72 rack uses a modular, cable-free tray design that enables faster assembly and serviceability, which also facilitates liquid cooling loop connections at rack scale.
  **Evidence:** "The rack introduces modular, cable-free tray designs for 18x faster assembly and serviceability versus NVIDIA Blackwell, combined with intelligent resiliency and software-defined NVLink routing, which ensures continuous operation and reduces maintenance overhead."
  **Category:** rack architecture
  **Relevance:** The shift to modular, cable-free tray designs implies that liquid cooling loops are also simplified and standardized at the rack level, reducing the complexity of connecting and disconnecting coolant plumbing during maintenance.
  **Confidence:** medium

### Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf

- **Evidence ID:** E007
  **Claim:** Extreme co-design across compute, networking, power delivery, and cooling is foundational to the Vera Rubin platform architecture, meaning cooling is no longer an afterthought but a co-designed system element.
  **Evidence:** "The NVIDIA Vera Rubin platform was designed for the shift in how intelligence is produced at scale, applying extreme co-design across compute, networking, power delivery, cooling, and system architecture to enable sustained intelligence production at AI factory scale."
  **Category:** cooling
  **Relevance:** Establishes that cooling is a first-class co-design consideration at the platform level, implying that facility cooling infrastructure must be planned and scaled in concert with compute and networking infrastructure.
  **Confidence:** high
- **Evidence ID:** E008
  **Claim:** AI factory racks must operate within tightly constrained power and cooling limits, implying that cooling infrastructure capacity is a hard constraint on AI factory scale-out.
  **Evidence:** "there is relentless demand to extend rack-scale performance while maintaining data-center-scale determinism within tightly constrained power and cooling limits."
  **Category:** cooling
  **Relevance:** Directly states that power and cooling limits are hard constraints on AI factory rack deployments, implying facilities must provision sufficient cooling capacity to match rack-scale power densities.
  **Confidence:** high

### Updates from NVIDIA GTC 2025 Conference.pdf

- **Evidence ID:** E009
  **Claim:** The NVIDIA roadmap shows NVL72 racks from Rubin onward are all liquid-cooled, indicating liquid cooling is the permanent infrastructure standard for future AI factory racks.
  **Evidence:** "Rubin NVL72 Liquid Cooled ... Kyber NVL576 Liquid Cooled"
  **Category:** cooling
  **Relevance:** The multi-year roadmap explicitly labels both the Rubin NVL72 and the next-generation Kyber NVL576 as liquid cooled, confirming that liquid cooling is the long-term infrastructure standard for AI factory racks and not a transitional measure.
  **Confidence:** high
- **Evidence ID:** E012
  **Claim:** Co-packaged silicon photonics networking switches save megawatts of power at AI factory scale, implying that reduced networking heat loads complement the shift to liquid cooling for compute.
  **Evidence:** "Co-packaged silicon photonics networking switches to scale AI factories to millions of GPUs ... ~3X the GPUs at ISO Power ... 3.5X Power efficiency"
  **Category:** cooling
  **Relevance:** Shows that power efficiency gains from co-packaged optics reduce aggregate thermal load in AI factories, which indirectly eases the burden on liquid cooling infrastructure by limiting heat generation in the networking layer.
  **Confidence:** medium

### NVIDIA GTC 2026_ Rubin GPUs, Groq LPUs, Vera CPUs, and What NVIDIA Is Building for Trillion-Parameter Inference - StorageReview.com.pdf

- **Evidence ID:** E011
  **Claim:** The LPX rack, housing 256 Groq 3 LPUs for decode acceleration, is fully liquid-cooled and built on MGX infrastructure, extending liquid cooling requirements to inference-specific racks.
  **Evidence:** "The LPX Rack is fully liquid-cooled, built on MGX infrastructure, and will be available in the second half of 2026, coincident with the broader Vera Rubin rollout."
  **Category:** cooling
  **Relevance:** Confirms that liquid cooling requirements extend beyond GPU compute racks to specialized inference accelerator racks (LPX), meaning all rack types in an AI factory deployment require liquid cooling infrastructure.
  **Confidence:** high

### NVIDIA Corporation - NVIDIA Kicks Off the Next Generation of AI With Rubin — Six New Chips, One Incredible AI Supercomputer.pdf

- **Evidence ID:** E014
  **Claim:** Lenovo is integrating its Neptune liquid-cooling solution with NVIDIA Rubin platform racks for enterprise AI factory deployments.
  **Evidence:** "Lenovo is embracing the next-generation NVIDIA Rubin platform, leveraging our Neptune liquid-cooling solution as well as our global scale, manufacturing efficiency and service reach, to help enterprises build AI factories that serve as intelligent, accelerated engines for insight and innovation."
  **Category:** cooling
  **Relevance:** Confirms that OEM partners are developing proprietary liquid-cooling solutions (e.g., Lenovo Neptune) specifically to support AI factory rack deployments, indicating that the broader data center supply chain is adapting to liquid cooling requirements.
  **Confidence:** high

## Evaluation Warnings

- None.
