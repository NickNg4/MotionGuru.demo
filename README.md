# MotionGuru — End-to-End ML Sports Analytics Pipeline

## Overview

MotionGuru is a fully automated sports analytics system that transforms raw, unstructured broadcast boxing footage into structured fighter dossiers containing exploitable behavioral patterns, biomechanical tendencies, and rule-based tactical game plans. Built entirely in Python as a modular, multi-phase pipeline, the system operates without any manual intervention between input (a YouTube link or video file) and output (a complete scouting report with actionable counter-strategies).

The project demonstrates end-to-end ownership of the full machine learning lifecycle: from raw data acquisition and computer vision preprocessing, through custom model architecture design and active-learning annotation, to deterministic profiling and evidence-based strategic reasoning.

---

## Demo

Watch the full pipeline in action — from raw broadcast footage to structured scouting report:

**[MotionGuru Demo Video](https://www.youtube.com/watch?v=K5-CoHNZcow)**

---

## Why This Matters — Superhuman Analysis

Traditional fight analysis relies on a human coach watching footage and forming subjective impressions. MotionGuru operates on a fundamentally different plane:

- **Sees what human eyes cannot.** The system detects sub-300ms preparatory tells — involuntary micro-movements like a lead-foot lift before an attack, or a rear elbow dipping toward the hip before a cross — that are invisible at real-time playback speed. These biomechanical "telegraphs" create exploitable counter-timing windows that no human coach could consistently identify from broadcast footage.
- **Quantifies degradation over time.** Instead of a coach's subjective sense that a fighter "looked tired in the later rounds," MotionGuru computes precise round-over-round decay slopes: guard height dropping X% per round, punch return paths looping wider, combination lengths shortening, flat-footed percentage increasing. These trajectories turn fatigue from a vague observation into a predictive signal — projecting *when* a fighter's structure will collapse and which specific vulnerabilities will open.
- **Exploits rhythm and timing patterns.** The system maps inter-punch intervals, between-combo latencies, and round pacing distributions at millisecond resolution. A metronomic rhythm (low variance in timing) is an exploitable pattern; MotionGuru identifies it quantitatively and maps it to specific counter-timing strategies.
- **Predicts future behavior from structural signatures.** By cross-referencing 107 behavioral metrics across 15 dimensions, the system identifies compound vulnerability patterns — e.g., a fighter who retreats in straight lines, drops their guard after punching, AND has a declining reaction time slope is not just individually exploitable on each axis, but catastrophically vulnerable to a specific combined pressure strategy. The profiler surfaces these intersections that no single human observation could assemble.

---

## Pipeline Architecture

```
Raw broadcast video
  → Phase 1a: Automated round segmentation via OCR-based timer detection
  → Phase 1b: Multi-fighter pose extraction with persistent identity tracking
  → Phase 2:  Biomechanical feature engineering (57 frame-level features)
  → Phase 3a: Temporal action classification (per-frame, per-fighter)
  → Phase 3b: Multi-dimensional fighter profiling (107 behavioral metrics)
  → Phase 3c: Evidence-based strategy generation with counter-technique mapping
```

---

## Phase 1 — Video Preprocessing & Pose Extraction

### Round Segmentation
- Automatically detects round boundaries from on-screen broadcast timers using OCR
- Extracts individual rounds via lossless stream-copy (no video re-encoding)
- Handles rest periods, stoppages, and replay segments

### Pose Extraction & Identity Tracking
- **Object Detection:** YOLO11m-pose for unified person detection and 17-keypoint pose estimation (with YOLOv8 fallback), plus one synthetic landmark for anatomical completeness (18 total keypoints per fighter per frame)
- **Re-Identification:** Deep ReID (OSNet) embeddings maintain consistent fighter identity across camera cuts, zoom changes, and occlusions
- **Identity Reconciliation:** A Viterbi dynamic programming pass post-processes the identity assignments to resolve transient swaps caused by fighters crossing paths or clinching
- **Signal Filtering:** Per-keypoint adaptive filtering with motion-aware parameters ensures fast limb movements (punches) are preserved while stable landmarks (hips, shoulders) are smoothed for noise reduction
- **Occlusion Handling:** Kalman filter-based identity recovery bridges short tracking gaps without corrupting the feature signal

**Output:** Normalized keypoint coordinate arrays per fighter per frame with associated confidence scores, plus annotated skeleton-overlay video for visual validation

---

## Phase 2 — Biomechanical Feature Engineering

Converts raw 2D keypoint coordinates into **57 biomechanical features per frame**, encoding domain-specific knowledge about boxing mechanics:

**Per-Fighter Features (26 each):**
- Limb kinematics (wrist speed, acceleration, elbow extension velocity and angle)
- Upper body mechanics (shoulder and hip rotation, rotation velocity)
- Guard geometry (wrist-to-face distances for lead, rear, and combined guard openness)
- Head movement dynamics (velocity and acceleration vectors for slip/duck detection)
- Postural indicators (torso lean angle, lean rate, stance width, center of mass velocity)
- Punch alignment metrics (hip-shoulder-elbow kinetic chain alignment)

**Cross-Fighter Interaction Features (4):**
- Inter-fighter distance and closing/separating rate
- Strike proximity metrics (wrist-to-target distances)

**Data Quality Channel (1):**
- Binary frame validity indicator preserving pose confidence information for downstream models

Features are z-score normalized per video, with the validity channel excluded from normalization to preserve its binary semantics. Invalid keypoint frames are handled via bounded forward-fill to prevent pose-at-origin artifacts from corrupting the feature signal.

---

## Phase 3a — Temporal Action Classification

### Model Architecture
- **Bidirectional Temporal Convolutional Network (TCN)** with symmetric (non-causal) dilations, allowing each frame to incorporate equal temporal context from past and future
- **Temporal Attention** mechanism (transformer encoder layer) on top of the convolutional backbone for capturing long-range dependencies
- **Dual output heads** — one per fighter — sharing a common backbone that learns cross-fighter interaction patterns (e.g., one fighter's offense correlating with the other's defensive reaction)
- Three-class classification: idle, offense, defense

### Training Pipeline
- Custom focal loss with label smoothing, addressing severe class imbalance (~85% idle frames)
- AdamW optimizer with cosine annealing learning rate schedule
- Skip and clinch zones excluded from loss computation via ignore masking, preventing corrupted tracking periods from polluting the training signal

### Active-Learning Annotation System
- Custom-built OpenCV annotation tool with frame-accurate playback, multi-speed review, and keyboard-driven labeling workflow
- **Review mode** enables rapid model-assisted labeling: the TCN predicts labels, the human confirms or rejects, and corrections feed back into the next training cycle
- Supports iterative label → train → predict → review → retrain cycles for continuous model improvement

### Inference Post-Processing
- Multi-stage pipeline: temporal smoothing → confidence thresholding → minimum duration filtering → same-class segment merging → gradient-based boundary refinement

---

## Phase 3b — Multi-Dimensional Fighter Profiling

A deterministic profiling engine that computes **107 behavioral and biomechanical metrics** per fighter per round, organized across **15 analytical dimensions**:

| # | Dimension | Category | Example Metrics |
|---|---|---|---|
| 1 | Punch Selection & Volume | Offensive | Hand dominance, workload per round, singles vs. combinations ratio |
| 2 | Combination Architecture | Offensive | Average combo length, pattern diversity and predictability |
| 3 | Preparatory Tells | Micro-Pattern | Pre-attack weight shifts, micro-steps before offense |
| 4 | Punch Biomechanics | Mechanical | Arc curvature, extension ratio, retraction speed |
| 5 | Guard Position & Habits | Defensive | Resting guard height, guard drop after punching, recovery time |
| 6 | Defensive Actions | Defensive | Reaction time, counter-after-defense rate, style classification |
| 7 | Retreat & Escape Geometry | Movement | Straight vs. lateral retreat ratio, pivot frequency |
| 8 | Advance & Pressure Patterns | Movement | Closing method, aggression scoring, post-exchange behavior |
| 9 | Stance & Foot Orientation | Structural | Stance identification, switch frequency, flat-footed percentage |
| 10 | Timing & Rhythm | Temporal | Inter-punch intervals, round pacing distribution |
| 11 | Pressure Response | Behavioral | Default behavior under aggression, punch degradation under pressure |
| 12 | Clinch & Inside Fighting | Behavioral | Initiation rate, post-break vulnerabilities |
| 13 | Fatigue & Biomechanical Decay | Temporal | Round-over-round decay slopes for punch rate, guard height, combo length |
| 14 | Psychological & Damage Tells | Micro-Pattern | Mental reset tics, damage compensation indicators |
| 15 | Kinematics & Center of Mass | Physics | Over-lean on punches, exchange-phase velocity, balance metrics |

Each metric carries a **confidence tag** (direct measurement, 2D proxy estimate, or requires additional data modality) ensuring downstream consumers know exactly how reliable each data point is.

**Cross-fight aggregation:** A roster-based identity resolution system links fighters across multiple bouts. Per-round profiles are merged into combined fighter dossiers using weighted averaging (numerics), logical OR with agreement tracking (booleans), and mode selection (categoricals).

---

## Phase 3c — Evidence-Based Strategy Engine

The strategy layer maps profiled fighter weaknesses to specific, actionable counter-techniques:

- **Rule-based matching:** A threshold rules engine evaluates each profiled metric against empirically calibrated thresholds derived from the analyzed fight dataset
- **Strategy library:** A curated library of tactical counter-strategies, each containing specific techniques, biomechanical reasoning for why the counter works, risk assessment, training drills, and difficulty ratings
- **Game plan generation:** Critical and high-priority findings are compiled into a prioritized tactical game plan with concrete training recommendations
- **Confidence-aware reporting:** Reports from insufficient data are flagged and game plan sections include reliability warnings

---

## Technical Highlights

- **No LLM dependency for analysis:** All profiling and strategy generation is purely deterministic Python computation, ensuring reproducibility and auditability. LLMs are optionally available only for prose formatting of final reports.
- **Domain-driven feature engineering:** 57 hand-crafted features encode boxing-specific biomechanical knowledge (kinetic chain mechanics, guard theory, pressure/retreat dynamics) rather than relying on generic learned representations.
- **Graceful degradation:** The system explicitly tracks what it can and cannot measure from 2D broadcast footage. Metrics requiring 3D depth data or granular action sub-types are marked as unavailable rather than silently approximated.
- **Production-hardened:** Comprehensive audit and bug-fix pass covering signal integrity, loss function mathematics, augmentation correctness, and metric computation accuracy across all pipeline stages.

---

## Full-Stack Application Architecture

The machine learning pipeline is wrapped in a production-ready, cloud-native web application, transforming the raw analysis engine into an interactive coaching dashboard.

### Frontend Application
Built with **React 19, TypeScript, and Vite**, leveraging **Zustand** for global state and **TanStack Query** for robust server-state synchronization. The interface is styled with **Tailwind CSS** and is broken down into purpose-built feature modules:

- **Authentication & Security (`auth`):** Secure, token-based authentication flow for user and role management.
- **Coaching Dashboard (`dashboard`):** A centralized command center surfacing recent analytics, high-priority strategic alerts, and aggregated fighter progression metrics.
- **Video Management (`videos`):** Interface for uploading 4K broadcast footage directly to S3/MinIO, managing video libraries, and triggering the asynchronous ML extraction pipeline.
- **Roster Management (`athletes` & `opponents`):** Distinct tracking systems for internal athletes verses scouted opponents, resolving identities across multiple bouts to aggregate career-long behavioral profiles.
- **Analysis View (`analysis`):** The core interactive tool for reviewing ML outputs. Features timeline-synced video playback overlaid with biomechanical extraction data and frame-accurate event tagging.
- **Tactical Reports (`reports`):** Consumer-facing presentation of the Phase 3c Strategy Engine. Displays the prioritized tactical game plans, specific counter-strategies, and generated behavioral summaries.
- **Testing:** Extensively tested with 120+ unit/integration tests via Vitest and React Testing Library, plus end-to-end validation using Playwright.

### Backend Services & Asynchronous Processing
- **RESTful API:** Powered by FastAPI (Python 3.11+) and SQLAlchemy, providing a high-performance asynchronous interface connecting the frontend to the ML engine.
- **Distributed Task Queue:** Heavy computer vision and ML profiling tasks are offloaded to Celery workers using Redis as the message broker, preventing the API from blocking during gigabyte-scale video processing.
- **Data Persistence:** 
  - **PostgreSQL + TimescaleDB** handle complex time-series telemetry data (18 keypoints × 57 features × thousands of frames per round).
  - **MinIO (S3-Compatible)** provides scalable object storage for raw 4K broadcast footage, segmented round clips, and annotated video outputs.

### Infrastructure & DevOps
- **Containerization:** Fully dockerized services orchestrated via Docker Compose for a seamless local development environment.
- **Infrastructure as Code (IaC):** Production environments are provisioned declaratively using Terraform, ensuring reproducible cloud deployments.
- **CI/CD:** Automated GitHub Actions workflows enforce code quality and run the 325+ combined test suite securely on every commit.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Zustand, TanStack Query |
| **Backend API** | FastAPI, Python 3.11+, SQLAlchemy, Celery |
| **Databases** | PostgreSQL 15, TimescaleDB, Redis, MinIO / S3 |
| **Computer Vision** | YOLO11m-Pose (YOLOv8 fallback), OSNet (ReID), EasyOCR, OpenCV |
| **Deep Learning** | PyTorch (custom TCN + Temporal Attention architecture) |
| **Testing** | Pytest (200+ tests), Vitest (120+ tests), Playwright |
| **DevOps / Infra** | Docker, Terraform, GitHub Actions |

---

## Planned Features

- **Automated Habit Clip Extraction:** Automatically identify and extract short video clips showcasing a fighter's top N best and worst habits — for both your own athlete and the scouted opponent. Each clip will be annotated with on-screen commentary explaining the detected pattern, why it matters tactically, and the specific counter-strategy or correction recommended. This transforms the raw scouting report from a data table into a visual coaching tool: a coach can review a fighter's "Top 5 Exploitable Habits" reel before a bout, or show an athlete their own "Top 5 Habits to Fix" with frame-by-frame evidence.
- **Granular action sub-classification (Phase 3b TCN):** Expand from 3 classes to 12+ (jab, cross, hook, uppercut, body shot, block, slip, duck, parry, clinch), unlocking ~40% of currently dormant profiler metrics including combination predictability scoring.
- **Multi-sport scalability roadmap:** The core architecture (pose → features → profiler → strategy) is sport-agnostic by design. The scaling path:
  1. **Boxing** (current) — proof of concept with the deepest domain model.
  2. **All combat sports** — MMA, kickboxing, Muay Thai, wrestling. Same skeleton tracking; swap the feature set and strategy library per discipline.
  3. **Individual sports** — Tennis (stroke mechanics, court positioning), fencing (blade work, footwork tells), golf (swing kinematics). One-vs-one or solo athlete analysis using the same profiler-to-strategy pipeline.
  4. **All broadcast sports** — Any sport where biomechanical patterns drive tactical advantage. The modular pipeline means each new sport requires only a new feature layer and strategy library, not a new system.
- **Predictive modeling layer:** Use accumulated cross-fight profiler data to build predictive models for fight outcome probability, round-by-round scoring projections, and optimal strategy selection based on opponent profile similarity matching.

---

## Status

- **Phase 1 (Pose Extraction):** Production-ready with robust multi-fighter tracking
- **Phase 2 (Feature Engineering):** Complete — 57-feature output with data quality tracking  
- **Phase 3a (Action Detection):** Active development — model training and active-learning labeling cycle in progress
- **Phase 3b (Fighter Profiling):** Implemented — 107 metrics across 15 dimensions with cross-fight aggregation
- **Phase 3c (Strategy Engine):** Implemented — threshold-based scouting reports with tactical game plans

---
## Contributor
- Hieu (Nick) Nguyen
- Vinh Ngo Gia
