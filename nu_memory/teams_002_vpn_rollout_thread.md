# Teams thread: #network-services — VPN standardization discussion

**Channel:** #network-services · *(Simulated thread for demo purposes)*

**Network Services lead:** Proposal doc is up for consolidating remote access on GlobalProtect. Key drivers: MFA integration out of the box, clients for every platform we support, and one portal (vpn.northeastern.edu) instead of per-department appliances.

**Service Desk manager:** Support angle — half our remote-access tickets are people confused about which VPN to use. One client, one set of instructions, one knowledge-base article. Strong yes from us.

**InfoSec analyst:** Security assessment complete. GlobalProtect passes baseline; requiring MFA enrollment before first connection closes the credential-theft gap for off-campus access. One note: cloud services like Microsoft 365 travel outside the tunnel by design — that's fine, the VPN is for protected NUnet resources.

**Network Services lead:** Correct — split-tunnel by design. Adding that to the FAQ so nobody expects the VPN to change their Office experience.

**Service Desk manager:** Flagging for the architecture review. If approved, we'll pre-write the install guides for Windows/macOS/Linux/mobile before announcement.