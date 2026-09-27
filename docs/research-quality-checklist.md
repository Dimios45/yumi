# What “usable UMI data” means for this custom rig

The supplied [UMI paper](https://arxiv.org/html/2402.10329v3) motivates synchronized wrist RGB, six-degree-of-freedom tool pose and jaw opening, sufficient visual context, train/deployment camera geometry consistency, robot-feasible trajectories, and measured observation/execution latency. Its action horizon is relative to the current tool pose, not a sequence of independently accumulated deltas.

Our exporter stores reconstructible episode-frame poses and next-frame targets. For a policy window anchored at t, derive each future transform as `inverse(T_episode_tcp(t)) @ T_episode_tcp(t+k)`. The existing dataset schema is not by itself the complete UMI policy/deployment interface. Robot-side action timing, matching camera placement and reachability must be checked for the intended robot.

The supplied [KIWI paper](https://arxiv.org/html/2609.22809) separates manipulation views from localization views, reconstructs hands in a shared frame and treats camera-to-tool geometry explicitly. Its camera-only 360-degree reconstruction cannot simply be transplanted into our D405/T265 setup. The [project page](https://lingfeng.moe/KIWI/) currently labels its code “coming soon”; no unpublished KIWI implementation or reported accuracy is claimed here.

| Requirement | Current rig status |
|---|---|
| Reproducible capture / real LeRobot v3 | Implemented; synthetic full readback tested |
| Wrist RGB and raw depth | Recorded; real frame loss still needs correction |
| Full-rate pose + IMU | Recorded without counter gaps in both supplied attempts |
| Reliable pose quality | Second attempt reached confidence 3 only during final 3.75 s |
| Metric opening | 10 mm marker size entered; actual jaw mapping and held-out validation pending |
| Metric TCP | Camera/tracker and camera/tool calibration pending |
| Synchronized streams | SDK-global clock handling corrected; residual timing calibration pending |
| Stable marker geometry | Most fits exceed current reprojection threshold; visibility/flatness/intrinsics need checking |
| Robot transfer | Target robot/camera geometry, action latency and reachability not yet specified/validated |
| Bimanual shared frame | Not implemented; two unrelated T265 origins would not suffice |
| Physical accuracy | Must be established with independent references; no blanket millimetre claim |

LeRobot v3 validates storage and indexing; it does not validate these physical requirements.
