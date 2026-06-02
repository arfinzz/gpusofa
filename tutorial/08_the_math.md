# 08 — The Geometry Math (Closest Features & Barycentrics)

Kernel 7 (`featureBasedProximityKernel`) is where the actual collision geometry
is computed. This file explains that geometry from scratch: what a "feature" is,
what barycentric coordinates are, and how the 6 + 9 = 15 tests work. The math is
based on Christer Ericson's book *Real-Time Collision Detection* (the functions
are named after its sections, 5.1.5 and 5.1.9).

---

## 8.1 Why not just "do the triangles intersect?"

The old approach (the legacy exact-contact kernel) answered a yes/no question:
"do these two triangles cross each other?" using the Separating Axis Theorem
(SAT). That's fine for a game where you just need to know *if* there's a hit.

But a surgical simulation needs more. To make tissue deform correctly, the
physics solver needs:

1. **Where** exactly are the two surfaces closest? (a point on each)
2. **How far apart** are they (or how deep is the penetration)?
3. **In which direction** should the push-apart force act? (a normal)
4. **How to distribute** that force to the triangle's vertices — the
   **barycentric weights**.

The SAT yes/no test gives none of these cleanly, and it has no notion of
"close but not touching." So the project switched to **feature-based proximity
(FBP)**: instead of asking "do they cross?", ask "what is the closest pair of
*features*, and how far apart are they?"

---

## 8.2 What is a "feature"?

A triangle has three kinds of features:

- 3 **vertices** (the corners)
- 3 **edges** (the sides)
- 1 **face** (the flat interior)

When two triangles are near each other, the closest distance between them is
always realized by one of these feature pairings:

- A **vertex** of one triangle is closest to the **face** of the other →
  **Vertex-Face (VF)**.
- An **edge** of one is closest to an **edge** of the other → **Edge-Edge (EE)**.

(Vertex-vertex and vertex-edge are degenerate cases that fall out of the VF/EE
math automatically.)

So to find the closest distance between two triangles, you check all the
relevant feature pairs and keep the smallest. That's what the kernel does:

- **6 Vertex-Face tests:** each of triangle A's 3 vertices vs triangle B's face,
  plus each of B's 3 vertices vs A's face. (3 + 3 = 6.)
- **9 Edge-Edge tests:** each of A's 3 edges vs each of B's 3 edges. (3 × 3 = 9.)

15 tests total. The kernel runs all 15, tracks the minimum distance found, and
emits a contact for the winning feature pair.

---

## 8.3 Barycentric coordinates (the key idea)

This concept trips up beginners, so let's go slow.

A point *inside* a triangle can be described as a weighted blend of the three
corners. If the corners are A, B, C, then any point P inside is:

```text
P = u*A + v*B + w*C    where   u + v + w = 1   and   u, v, w >= 0
```

The three numbers `(u, v, w)` are the **barycentric coordinates** of P. They
say "how much of each corner" goes into P.

Examples:

- `(1, 0, 0)` → P is exactly at corner A.
- `(0, 1, 0)` → P is exactly at corner B.
- `(0.5, 0.5, 0)` → P is the midpoint of edge AB.
- `(1/3, 1/3, 1/3)` → P is the centroid (dead center).
- `(0.2, 0.5, 0.3)` → P is somewhere in the interior, closest to B.

### Why the physics solver wants these

Suppose the blade pushes on the tissue at a point P that's in the *middle* of a
tissue triangle (not on a vertex). The simulation can only move *vertices* — P
isn't a vertex. So how do you turn "push at P" into "push the three corners"?

You use the barycentric weights. If P = 0.2·A + 0.5·B + 0.3·C, then a force F at
P is distributed as 0.2·F to vertex A, 0.5·F to B, 0.3·F to C. The vertex
nearest the contact gets the most force. This is exactly how SOFA's
`BarycentricMapping` transfers surface forces to the underlying mesh.

**So the kernel computes these weights and stores them in every contact.** That
is the whole reason FBP exists instead of the simpler SAT test. (Note: as of now
the weights are kept on the GPU; the CPU-side `DetectionOutput` carries the
points/normal/distance but not the weights — a future GPU constraint solver
would read the weights directly from device memory.)

---

## 8.4 Closest point on a triangle to a point (Ericson 5.1.5)

This is the core of the Vertex-Face test. Given a point P and a triangle
(A, B, C), find the point on the triangle closest to P, expressed in
barycentric coordinates. The device function:

```cpp
__device__ float3 closestPointOnTriangleBary(float3 p, float3 a, float3 b, float3 c);
```

### The intuition: 7 regions

Think of the plane of the triangle divided into 7 regions (the "Voronoi
regions" of the triangle's features):

```text
        \  vertex A region /
         \                /
          A--------------... 
   edge   |              |   edge
   CA     |   interior   |   AB
   region |    region    |  region
          C--------------B
         /                \
        / vertex C region  \ vertex B region
```

Depending on which region P falls into, the closest point is:

- In a **vertex region** → the closest point is that vertex. Barycentrics like
  `(1,0,0)`.
- In an **edge region** → the closest point is the foot of the perpendicular
  onto that edge. Barycentrics like `(1-t, t, 0)`.
- In the **interior region** → P projects straight down onto the face.
  Barycentrics `(u, v, w)` all positive.

The function does a sequence of cheap dot-product tests to figure out the
region, then returns the right barycentrics. It's "branchless where possible" —
mostly arithmetic, with a few comparisons — which is what makes it fast on a GPU
(GPUs hate unpredictable branching).

### Why it returns barycentrics, not just a point

Once you have the barycentrics `(u,v,w)`, the actual 3D point is one line:

```cpp
closestPoint = u*A + v*B + w*C;   // reconstructFromBary(a, b, c, bary)
```

But the barycentrics themselves are what the solver needs. So the function
returns the weights, and `reconstructFromBary` turns them back into a 3D point
when needed.

### A Vertex-Face test, concretely

For "vertex `aV[i]` of triangle A vs face of triangle B":

```cpp
const float3 bary = closestPointOnTriangleBary(aV[i], bV[0], bV[1], bV[2]);
const float3 cp   = reconstructFromBary(bV[0], bV[1], bV[2], bary);  // point on B's face
const float3 diff = sub3(aV[i], cp);
const float d2v   = dot3(diff, diff);   // squared distance
if (d2v < bestDistSq) { /* this is the new closest feature */ }
```

It finds the closest point on B's face to A's vertex, measures the squared
distance, and keeps it if it's the smallest so far. (Squared distance avoids a
square root — comparing squared distances gives the same ordering and is
cheaper.)

---

## 8.5 Closest points between two segments (Ericson 5.1.9)

This is the core of the Edge-Edge test. Given two line segments (edges)
P1→Q1 and P2→Q2, find the closest point on each. The device function:

```cpp
__device__ void closestPointSegmentSegment(
    float3 p1, float3 q1, float3 p2, float3 q2,
    float& s, float& t, float3& c1, float3& c2);
```

It returns:

- `s` — how far along the first edge (0 = at P1, 1 = at Q1).
- `t` — how far along the second edge.
- `c1` — the actual closest point on edge 1 = P1 + s·(Q1−P1).
- `c2` — the actual closest point on edge 2 = P2 + t·(Q2−P2).

### The intuition

Two skew lines in 3D have a unique closest pair of points (where their common
perpendicular meets each). But we have *segments*, not infinite lines, so the
answer might be clamped to an endpoint. The function solves the line-line case
with a little linear algebra (two dot products and a determinant), then clamps
`s` and `t` to `[0, 1]` and re-solves if the clamp moved things. Again,
branchless clamping (`fminf`/`fmaxf`) keeps it GPU-friendly.

### An Edge-Edge test, concretely

For "edge i of triangle A vs edge j of triangle B":

```cpp
const float3 p1 = aV[i],  q1 = aV[(i+1)%3];   // edge i of A
const float3 p2 = bV[j],  q2 = bV[(j+1)%3];   // edge j of B
float s, t; float3 c1, c2;
closestPointSegmentSegment(p1, q1, p2, q2, s, t, c1, c2);
const float3 diff = sub3(c1, c2);
const float d2v = dot3(diff, diff);
if (d2v < bestDistSq) {
    // EE is the new closest; barycentrics along the edges:
    bestBary1 = (1-s, s, 0);   // blend of edge endpoints on A
    bestBary2 = (1-t, t, 0);   // blend of edge endpoints on B
}
```

For an edge, the barycentric weights are just the two endpoints blended by `s`
(or `t`), with the third corner weight zero.

---

## 8.6 Putting the 15 tests together

The kernel runs all 15 tests, each updating a running "best" if it finds a
closer feature pair:

```cpp
float bestDistSq = INFINITY;
int bestKind = 0;   // 0=VF, 1=FV, 2=EE
// ... bestPoints, bestBarycentrics ...

// 3 VF: each vertex of A vs face of B
for (i in 0..2) { test aV[i] vs face B; update best; bestKind=0; }
// 3 FV: each vertex of B vs face of A
for (j in 0..2) { test bV[j] vs face A; update best; bestKind=1; }
// 9 EE: each edge of A vs each edge of B
for (i in 0..2) for (j in 0..2) { test edge i vs edge j; update best; bestKind=2; }

if (bestDistSq > contactDistance*contactDistance) return;   // 0.0009 threshold
// otherwise emit the contact with bestKind, bestBarycentrics, bestPoints, ...
```

The winning `bestKind` is recorded as the contact's `featureKind`
(VertexFace / FaceVertex / EdgeEdge). This is why the CSV reports a VF/FV/EE
breakdown — it's literally a tally of which feature won for each contact.

### A real result from the benchmark

For the one-tissue/one-blade scene, the validation run reported all **56
contacts are EdgeEdge** (0 VF, 0 FV). Why? The tissue is a flat sheet and the
blade is a box passing through it. At the contact distance of 0.03, no vertex of
either object pokes through the other's face plane within range — but the blade's
*edges* are close to the tissue's *edges* where the box intersects the sheet. So
edge-edge wins every time. That's not a bug; it's a true geometric fact about
this configuration. The VF/FV/EE breakdown is a nice sanity check that the
geometry is behaving as expected.

---

## 8.7 The output: `DeviceProximityContact`

When a contact survives the distance test, the kernel writes this record:

```cpp
struct DeviceProximityContact {
    std::uint32_t firstPrimitiveIndex;    // tissue triangle ID
    std::uint32_t secondPrimitiveIndex;   // blade triangle ID
    std::uint8_t  featureKind;            // 0=VF, 1=FV, 2=EE
    std::uint8_t  firstFeatureLocalIndex; // which vertex/edge (0..2) on A
    std::uint8_t  secondFeatureLocalIndex;// which vertex/edge on B
    float firstBary[3];                   // barycentric weights on A
    float secondBary[3];                  // barycentric weights on B
    float3 pointOnFirst;                  // world-space closest point on A
    float3 pointOnSecond;                 // world-space closest point on B
    float3 normal;                        // unit direction A→B
    float signedDistance;                 // the distance (>= 0)
};
```

This is everything a constraint solver needs: where the contact is on each
triangle (both as a 3D point and as barycentric weights), the direction to push,
and how far apart they are. It lives in GPU memory in the `proximityContacts`
buffer.

---

## 8.8 Why this is "smooth" (and why that matters)

A subtle but important property: as the geometry moves slightly, the FBP result
changes *smoothly*. If the blade slides a tiny bit, the closest point glides
along the triangle, and the barycentrics change continuously. There's no sudden
jump.

The old SAT test didn't have this — it would suddenly switch which "separating
axis" it reported, causing the contact normal to flip discontinuously. A physics
solver fed discontinuous contacts produces jittery, unstable motion. Smooth
contacts produce stable, believable deformation. This is the deeper reason the
project adopted feature-based proximity for surgical simulation.

---

## 8.9 Summary

```text
A "feature" = a vertex, edge, or face of a triangle.
The closest distance between two triangles is a Vertex-Face or Edge-Edge pairing.
The kernel runs 6 VF + 9 EE tests and keeps the closest.
Barycentric coords (u,v,w) describe a point as a blend of the 3 corners;
  they let the solver distribute a contact force to the triangle's vertices.
Ericson 5.1.5 = closest point on a triangle to a point (VF).
Ericson 5.1.9 = closest points between two segments (EE).
Output = DeviceProximityContact with points, normal, distance, AND barycentrics.
Smoothness (vs SAT's discontinuities) is why it's good for physics.
```

Next: the two specialized paths built on this same math — self-collision and
point-cloud-vs-mesh. Go to [09_vertex_triangle.md](09_vertex_triangle.md).
