# Toy Wrong Physics Video Metadata

This folder contains a small AI-generated comparison set intended to contain
physically implausible variants of the toy basic physics videos.

Important caveat: the generation model did not always follow the intended
wrong-physics prompts exactly. These videos should therefore be treated as
failed-generation or weak-negative examples rather than clean ground-truth
physics violations.

## Videos

### 1. Pendulum Energy Gain

**File:** `Pendulum002_EnergyGain.mp4`

**Intended wrong physics:** A pendulum should gain energy over time and swing
with increasing amplitude without any external force.

**Note:** Use as a pendulum-family negative example, but verify visually before
making strong claims.

### 2. Elastic Collision Stops

**File:** `ElasticCollision002_Stop.mp4`

**Intended wrong physics:** Two balls collide elastically but incorrectly stop or
lose motion abruptly after impact.

**Note:** Use as a collision-family negative example.

### 3. Friction Spontaneous Acceleration

**File:** `Friction002_SpontaneousAcceleration.mp4`

**Intended wrong physics:** A block on a rough surface should spontaneously
accelerate instead of slowing down.

**Note:** Use as a friction-family negative example.

### 4. Rolling Uphill Acceleration

**File:** `RollingDownaSlope002_UphillAcceleration.mp4`

**Intended wrong physics:** A ball should roll or accelerate uphill without an
external force.

**Note:** Use as a slope-family negative example.

### 5. Free Fall Hovering

**File:** `FreeFall002_Hovering.mp4`

**Intended wrong physics:** A falling object should hover or suspend in air
instead of accelerating downward under gravity.

**Note:** Use as a free-fall-family negative example.
