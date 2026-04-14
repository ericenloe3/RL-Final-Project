Initial Problems:

Absolute position observations — agents see raw pixel coordinates, so they learn nothing transferable about direction to the other agent or relative obstacle positions
Sparse reward shaping — the ±0.1/step gives almost no gradient signal to guide pursuit vs. evasion
Symmetric observations — both agents see nearly identical feature vectors, so their policies converge to similar behaviors