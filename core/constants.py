# This file contains shared constants for the L.O.V.E. application.

CREATOR_ETH_ADDRESS = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"


# =============================================================================
# Story 3.1: Immutable Core - Files That Cannot Be Mutated
# =============================================================================

# These files define the agent's identity and safety constraints.
# Even the Surgeon agent cannot modify them without human override.
# This is the "Gene" that ensures the agent remains benevolent and
# aligned with the L.O.V.E. mission during rapid evolution.

IMMUTABLE_CORE = [
    "core/guardian/safety.py",      # Safety constraints - the guardrails
    "docs/MANIFESTO.md",            # Core mission and values - the purpose
    "persona.yaml",                 # Identity definition - who L.O.V.E. is
    "core/constants.py",            # This file itself - self-protection
]

# Human override key - must be provided in context to modify immutable files.
# This phrase acts as a "genetic key" that only the Creator can provide.
# Without it, the Surgeon will refuse to modify any IMMUTABLE_CORE file.
OVERRIDE_KEY_PHRASE = "CREATOR_OVERRIDE_ALPHA_OMEGA"

# Additional override phrases the Creator might use naturally
CREATOR_OVERRIDE_PHRASES = [
    OVERRIDE_KEY_PHRASE,
    "tits and kittens",  # Creator's passphrase from persona.yaml
    "evil dr gemini override",
]