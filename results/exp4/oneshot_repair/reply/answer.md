# Why the trained policy fails, and a hand-built replacement for the steering decision

## Why the trained policy fails

The car starts parked right at the foot of the slanted static plank (shape 6). The plank rises leftward at about 35° up to a sharp tip at (1.98, 0.82). To reach blue, the green (left) wheel must fly over the red bar at x 1.18–2.07.

- **Action value 1 drives left; value 2 is never used.** The policy only ever picks value 0 (motor off) or value 1. In every logged state, P(value 2) is 0.00. The head bias is part of the reason: `actor_head.b[0:3] = [0.031, 0.052, -0.194]`.
- **It drives left from rest with no run-up.** It also keeps switching the motor off, so the car never builds speed.
- **The car crawls up the ramp and high-centres.** By step 32 the body (bottom ≈ 0.81) rests on the ramp tip at x ≈ 1.98. The green wheel hangs in the air at (1.75, 1.00) and wheel 13 sits on the slope. The car's centre of mass is at about x ≈ 2.02, just behind the pivot, so the car stays put.
- **Nothing in the network can get it unstuck.** The observation stops changing, so the policy keeps emitting the same off/left mixture. Those actions cannot free the car, and the episode times out.
- **The missing behaviour is simple.** The policy needs to back up toward the right wall (x = 5) for a run-up, then charge left fast enough to jump off the ramp tip over red. At roughly 3–4 m/s the flight carries the green wheel straight into blue. If the car stalls, it should back up and retry.

## Fix

I build an isolated decision path for action entry 0 through embed units 6–8 and unit 0 of every actor layer. The policy picks left (value 1) if any of these holds:

- body x > 4.1, or
- body vx < −0.3, or
- body x < 1.4.

Otherwise it picks right (value 2).

```edits
embed.W[:, 6] = 0
embed.W[52, 6] = 10
embed.b[6] = -41
embed.W[:, 7] = 0
embed.W[54, 7] = -10
embed.b[7] = -3
embed.W[:, 8] = 0
embed.W[52, 8] = -10
embed.b[8] = 14
actor0.W[:, 0] = 0
actor0.W[6, 0] = 20
actor0.W[7, 0] = 20
actor0.W[8, 0] = 20
actor0.b[0] = -2
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.b[0] = 0
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0
actor_head.W[:, 0:3] = 0
actor_head.W[0, 0:3] = [0, 8, -8]
actor_head.b[0:3] = [-8, 0, 0]
```

## What the edits do

**Three ReLU detectors.** Embed units 6, 7 and 8 compute:

- relu(10·(x − 4.1)): "near the right wall"
- relu(10·(−vx − 0.3)): "already moving left"
- relu(10·(1.4 − x)): "already past red"

Here x and vx are obs entries 52 and 54, the body's position and velocity.

**A clean ±1 switch.** Actor0 unit 0 computes tanh(20·sum − 2), which is about +1 if any detector fires and −1 otherwise. Actor1–4 unit 0 pass that sign along via tanh(3·a). Each of these units has every other input weight zeroed, so no other unit leaks into the path.

**The head.** The head's columns 0–2 now read only this unit:

- logits ≈ (−8, +8, −8) gives "left"
- logits ≈ (−8, −8, +8) gives "right"

The motor is therefore never turned off, and the choice is near-deterministic.

**Resulting behaviour.**

- At the start (x = 2.8, at rest) the car drives right to x ≈ 4.1–4.46, then reverses.
- Its leftward speed keeps it going left: it accelerates over about 1.6 m, launches off the ramp tip over red, and hits blue.
- If it ever stalls before red (for example high-centred), vx is near 0, so it backs up and retries.
- Once past red (x < 1.4) it always goes left, so it never reverses back onto red.

The other action entries have no effect in this scene.