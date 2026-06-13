"""Runtime configuration. CONFIRM_TX guards transmissions against agent
mistakes (wrong frequency/device) — not licensing enforcement. Licensed
users may set it to False."""

CONFIRM_TX: bool = True
