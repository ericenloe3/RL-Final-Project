The fix requires changes to all four files: relative/normalized observations in the env, progress-based reward shaping, a shared-trunk network in models, and online observation normalization in training.

Root cause: observations encoded the wrong things
The v1 observations gave agents [hunter_x, hunter_y, prey_x, prey_y, ...] — raw absolute pixel coordinates. An agent sitting at (400, 300) seeing prey at (600, 300) gets a feature vector of [400, 300, 600, 300, ...]. That looks nothing like the same situation with different starting coordinates, so the network couldn't generalise to "move toward the prey". The heading-to-prey direction was implicit and buried in a subtraction the network had to discover on its own.

Fix 1 — my_game_env.py: Relative, normalised observations
Each agent now sees (other_pos − own_pos) / [W, H] as an explicit feature. For the hunter this is a vector pointing toward the prey. For the prey it points toward the hunter, and the prey's reward is maximised by moving opposite to it. This asymmetry means the two policies are incentivised to behave differently from the very first gradient step. Obstacles are now sorted nearest-first (by surface distance, not centre distance) so index 0 is always the most immediately dangerous one.
Fix 2 — my_game_env.py: Dense progress + heading rewards
The v1 ±0.1/step gave no information about direction — an agent moving perpendicular to the prey got the same reward as one charging straight at it. The new rewards are:

Progress reward: R_PROGRESS_SCALE × (Δdist / AGENT_SPEED) — positive for the hunter when distance shrinks, mirror-negative for the prey. Scaled so a full-speed correct approach = +2.0/step instead of the old flat -0.1.
Heading bonus: R_HEADING_SCALE × dot(action, ideal_direction) — directly rewards each agent for moving in the right direction, even if the distance change is small that step (e.g. going around an obstacle).

Fix 3 — models.py: Shared trunk + RunningMeanStd
The actor and critic now share a two-layer MLP encoder before splitting. This halves the number of parameters while making critic gradients improve the actor's feature representation. The RunningMeanStd class does Welford online normalisation so observations passed to the network stay near N(0,1) throughout training.
Fix 4 — train.py + evaluate.py: Normalisation applied consistently
The normaliser is updated each rollout during training and its mean/var/count state is saved inside the .pt checkpoint. evaluate.py loads it back and applies normalizer.normalize() before every inference call — without this, the policy sees out-of-distribution inputs at test time even if training worked perfectly.