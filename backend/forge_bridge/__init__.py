"""
Forge Bridge — MiroFish → Forge Signal Integration

Translates swarm intelligence predictions from MiroFish multi-agent simulations
into quantified trading signals compliant with Forge v5.2 methodology.

Architecture:
    MiroFish (simulation) → Signal Bridge (this package) → PostgreSQL → Forge (validation)

Signal Classification (Forge §4.1): Informational
    - Arises from faster/better processing of public information via swarm intelligence
    - Randomised baseline: Mode A (kill gate)
    - n_free_parameters: 0 (no curve fitting)
"""

__version__ = "1.0.0"
__forge_methodology_version__ = "forge-v5.2-final"
