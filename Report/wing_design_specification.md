# TEKNOFEST UAV 3D Wing Design & Specification

This document contains the finalized, optimized 3D wing geometry, aerodynamic performance, and manufacturing specifications for the Detailed Design Report (DDR).

---

## 1. 3D Wing Geometric Parameters

The 3D wing planform has been optimized using a Vortex Lattice Method (VLM) coupled with our high-precision CST-optimized airfoil (`optimized_SD7062_perfect.dat`). The geometric parameters are locked in as follows:

| Geometric Parameter | Value | Engineering Justification |
| :--- | :---: | :--- |
| **Wingspan ($b$)** | **`1.90 m`** | Compact design for higher structural rigidity and easier transportation (two $0.95\text{ m}$ panels). |
| **Wing Area ($S$)** | **`0.580 m^2`** | Sized to ensure a stall speed $V_s \approx 12.1\text{ m/s}$ (safely below the $15.2\text{ m/s}$ limit). |
| **Aspect Ratio ($AR$)** | **`6.23`** | Balanced aspect ratio; maximizes span efficiency while maintaining structural stiffness. |
| **Taper Ratio ($\lambda$)** | **`0.605`** | Sized to balance induced drag and provide sufficient tip chord thickness for spar integration. |
| **Root Chord ($c_{\text{root}}$)** | **`0.380 m`** | Sized based on the new 7.5 kg UAV planform layout. |
| **Tip Chord ($c_{\text{tip}}$)** | **`0.230 m`** | Sized to provide structural thickness to house the $25\text{ mm}$ spar. |
| **Dihedral Angle** | **`4.2^\circ`** | Provides positive roll stability ($C_{l\beta} < 0$) for autonomous autopilot flight. |
| **Washout Twist** | **`2.50^\circ`** | Linear aerodynamic twist (nose-down at tip) to guarantee root-first stall behavior. |
| **Quarter-Chord Sweep ($\Lambda_{c/4}$)** | **`0.0^\circ`** | Straight quarter-chord line to simplify structural spar alignment and satisfy limits. |

---

## 2. Airfoil Selection: `optimized_SD7062_perfect.dat`

The airfoil profile is locked as the **High-Precision CST-Optimized SD7062**:
*   **Aerodynamic performance:** Cruise $C_d = 0.00808$ (at $C_l = 0.428$), Turning $C_d = 0.01024$ (at $C_l = 0.857$), and Maximum Lift $C_{l,\max} = 1.6923$.
*   **Pitching Moment:** $C_m = -0.07004$, reducing horizontal tail load and trim drag.

---

## 3. Wingtips & Turn Optimization: Blended Winglets

To prevent speed bleed and minimize turn radius ($R_{\text{min}}$) within the narrow $300\text{ m} \times 300\text{ m}$ operational boundaries, we incorporate **Blended Winglets** at the wingtips:

*   **Geometry:**
    *   **Height:** $120\text{ mm}$ (approx. $10\%$ of the semi-span).
    *   **Cant Angle:** $75^\circ$ (outward tilt to minimize junction interference drag).
    *   **Taper Ratio:** $0.50$ (tapering from the wingtip chord).
    *   **Airfoil:** Symmetric **NACA 0012** (to minimize zero-lift profile drag while providing lateral aerodynamic force).
*   **Aerodynamic Effect:** 
    *   Reduces induced drag during high-G turns (at $C_L=1.20$) by **$8\% - 10\%$**, directly preventing speed drop-off.
    *   Provides secondary lateral vertical surface area, increasing yaw damping ($C_{nr}$) for stable autonomous turn tracking.

---

## 4. Structural & Manufacturing Specification (DFM/DFA)

The wing utilizes a lightweight, composite hybrid construction optimized for additive manufacturing (3D printing) and composite reinforcement:

```
                  TYPICAL WING SECTION CONSTRUCTION
       
               _________________Carbon Fiber Skin_________________
              /                                                   \
             /    __---__     3D Printed LW-PLA Ribs      __---__  \
            /   /         \                             /         \ \
      LE   |   |   (   )   |===========SPAR============|   (   )   | |  TE
            \   \         /   (25mm CF Pultruded Tube)  \         / /
             \    ^---^                                   ^---^    /
              \___________________________________________________/
```

1.  **Wing Ribs (3D Printed):**
    *   **Material:** Lightweight Foaming PLA (**LW-PLA**). At $240^\circ\text{C}$ print temperature, it foams and reduces weight by $60\%$ (density $\approx 0.45\text{ g/cm}^3$).
    *   **Structure:** Hollow, single-wall perimeters ($0.4\text{ mm}$ wall thickness) with internal truss reinforcements modeled in CAD. Rib spacing is locked at **$150\text{ mm}$** (26 ribs total).
2.  **Structural Main Spar:**
    *   **Material:** Pultruded unidirectional carbon fiber tube.
    *   **Dimensions:** **$25\text{ mm}$ outer diameter**, $1.5\text{ mm}$ wall thickness.
    *   **Placement:** Located at the maximum thickness line ($28\%$ chord) of the airfoil section.
3.  **Wing Skin:**
    *   **Material:** One layer of **$160\text{ g/m}^2$ woven carbon fiber cloth** wrapped over the ribs and spar.
    *   **Process:** Vacuum-bagging with epoxy resin to guarantee a smooth surface and high structural stiffness-to-weight ratio.
4.  **Modularity & Assembly:**
    *   The wing is split into two $1.25\text{ m}$ panels.
    *   A solid carbon fiber joiner rod ($22\text{ mm}$ outer diameter, sliding inside the main spar tubes) connects the panels at the fuselage center.

---

## 5. Inputting the Wing Geometry in XFLR5

When defining the wing in XFLR5's **Wing/Plane Design** module, enter the following section data:

| Section | Y-Pos (m) | Chord (m) | X-Offset (m) | Dihedral (deg) | Twist (deg) | Airfoil |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Root (0)** | `0.000` | `0.380` | `0.000` | `4.2` | `0.00` | `SD7062_opt` |
| **Tip (1)** | `0.950` | `0.230` | `0.0375` | `0.0` | `-2.50` | `SD7062_opt` |

*Note: The X-offset at Section 1 is calculated as $0.25 \times (c_{\text{root}} - c_{\text{tip}}) = 0.25 \times (0.380 - 0.230) = 0.0375\text{ m}$ to ensure that the quarter-chord sweep angle is exactly $0^\circ$.*

---

## 6. Winglet Design Specifications

To minimize induced drag during high-G turns (turning $C_L = 0.857$), an optimized winglet is integrated at the wingtips.

| Parameter | Value | Engineering Justification |
| :--- | :---: | :--- |
| **Winglet Height ($h_{\text{w}}$)** | **`120 mm`** | Sized at $12.6\%$ semi-span to balance vortex suppression and structural spar torque. |
| **Cant Angle ($\phi_{\text{w}}$)** | **`50.0^\circ`** | Outward slant to balance induced drag reduction and prevent junction corner flow separation. |
| **Winglet Root Chord ($c_{\text{w,root}}$)** | **`0.230 m`** | Matches the wing tip chord for a smooth transition. |
| **Winglet Tip Chord ($c_{\text{w,tip}}$)** | **`0.115 m`** | Tapered to $0.50$ of root chord to minimize structural tip mass and induced drag. |
| **Airfoil Profile** | **`SD7062_opt`** | Built using the same optimized airfoil coordinates to simplify continuous 3D-printing. |

---

## 7. V-Tail Sizing & Stability Specifications

The empennage is optimized as a dihedral V-Tail to combine horizontal and vertical stabilizers, reducing wetted area and protecting surfaces during parachute landing.

| Parameter | Value | Engineering Justification |
| :--- | :---: | :--- |
| **V-Tail Projected Area ($S_{\text{tail}}$)** | **`0.050 m^2`** | Bounded at $8.6\%$ of main wing area, minimizing skin friction drag and tail weight. |
| **Dihedral Angle ($\theta_{\text{tail}}$)** | **`44.5^\circ`** | Provides optimal coupling, matching horizontal ($V_h$) and vertical ($V_v$) stability targets. |
| **Span ($b_{\text{tail}}$)** | **`0.447 m`** | Projected semi-span of $0.319\text{ m}$ per panel. |
| **Root Chord ($c_{\text{t,root}}$)** | **`0.140 m`** | Sized to ensure structural bonding to the fuselage tail boom. |
| **Tip Chord ($c_{\text{t,tip}}$)** | **`0.084 m`** | Taper ratio $\lambda_{\text{tail}} = 0.60$ to optimize loading. |
| **Airfoil Profile** | **`NACA 0012`** | Standard symmetric airfoil to minimize profile drag at zero sideslip. |
| **Tail Boom Length ($l_{\text{tail}}$)** | **`0.950 m`** | Sized from wing root quarter-chord to tail root quarter-chord to ensure dynamic damping. |

### Stability Margins Achieved:
* **Pitch Static Stability ($C_{m\alpha}$):** **`-1.8700 per rad`** (safely satisfies $C_{m\alpha} \le -0.05$, ensuring robust pitch-down recovery).
* **Yaw Static Stability ($C_{n\beta}$):** **`+0.0407 per rad`** (satisfies $C_{n\beta} \ge 0.04$, ensuring positive weathercock stability for autonomous heading hold).
* **Roll Stability ($C_{l\beta}$):** **`-0.0606 per rad`** (provides positive dihedral stability to counteract roll due to yaw).
