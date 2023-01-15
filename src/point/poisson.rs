//
// Copyright 2020 Hans W. Uhlig. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

use rand::prelude::{Rng, SliceRandom};
use std::hash::{Hash, Hasher};

/// Result of [`PoissonDiscSampler`] Sampling Step
pub enum PoissonEvent {
    /// Created Origin Point
    Origin { point: PoissonPoint },
    /// Created New Point for sampling Children.
    Created {
        parent: PoissonPoint,
        point: PoissonPoint,
    },
    /// Point will no longer sample children.
    Closed { point: PoissonPoint },
    /// Generation is complete
    Complete,
}

/// Point generated from a [`PoissonDiscSampler`]
#[derive(Copy, Clone, Debug, Default)]
pub struct PoissonPoint {
    pub id: u64,
    pub x: f64,
    pub y: f64,
}

impl Hash for PoissonPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq for PoissonPoint {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl std::fmt::Display for PoissonPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}: {}, {})", self.id, self.x, self.y)
    }
}

/// Poisson Disc Sampler
///
/// This is an implementation of a Poisson Disc Sampler using an improved version of Bridson's
/// Algorithm by Martin Roberts as detailed [here](https://observablehq.com/@techsparx/an-improvement-on-bridsons-algorithm-for-poisson-disc-samp/2).
///
/// It Should generate a tree of points cascading out from the central point at a general radius.
///
/// ```rust
/// use progen::point::{PoissonDiscSampler, PoissonEvent, PoissonPoint};
/// use rand::{SeedableRng, rngs::SmallRng};
/// use std::collections::HashMap;
///
/// let mut sampler = PoissonDiscSampler::new(100, 100, 2.0, SmallRng::from_entropy());
/// let mut nodes: HashMap<u64, PoissonPoint> = HashMap::default();
/// let mut edges: Vec<(u64, u64)> = Vec::default();
/// loop {
///   match sampler.step() {
///     PoissonEvent::Origin { point } => {
///       println!("Origin Point: {}", &point);
///       nodes.insert(point.id, point);
///     }
///     PoissonEvent::Created { parent, point } => {
///       println!("Created Point: {}", &point);
///       nodes.insert(point.id, point);
///       edges.push((parent.id, point.id));
///     }
///     PoissonEvent::Closed { point } => {
///       println!("Closed Point: {}", &point);
///     }
///     PoissonEvent::Complete => break,  
///   }
/// }
/// println!("Poisson Disk Sample Graph: Nodes({}), Edges({})", nodes.len(), edges.len());
/// ```
///
#[derive(Clone)]
pub struct PoissonDiscSampler<R: Rng> {
    /// Round Counter
    round: u64,
    /// Width of "canvas"
    width: u64,
    /// Height of Canvas
    height: u64,
    /// Radius of point padding
    radius: f64,
    /// Square of Radius
    radius2: f64,
    /// Size of Grid Cell
    cell_size: f64,
    /// Width in Grid Cells
    grid_width: u64,
    /// Height in Grid Cells
    grid_height: u64,
    /// Max Samples (k) before Rejection
    max_samples: usize,
    /// Grid of Cells
    grid: Vec<Option<PoissonPoint>>,
    /// Queue of Open Points
    queue: Vec<PoissonPoint>,
    /// Pseudo Random Number Generator
    rng: R,
}

impl<R: Rng> std::fmt::Debug for PoissonDiscSampler<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoissonDiscSampler")
            .field("round", &self.round)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("radius", &self.radius)
            .field("cell_size", &self.cell_size)
            .field("grid_width", &self.grid_width)
            .field("grid_height", &self.grid_height)
            .field("max_samples", &self.max_samples)
            .field("queue_size", &self.queue.len())
            .finish()
    }
}

impl<R: Rng> PoissonDiscSampler<R> {
    #[tracing::instrument(skip(rng))]
    pub fn new(width: u64, height: u64, radius: f64, rng: R) -> PoissonDiscSampler<R> {
        let max_samples = 4; // maximum number of samples (k) before rejection
        let round = u64::MAX;
        let radius2 = radius * radius;
        let cell_size = radius * f64::sqrt(1.0 / 2.0);
        let grid_width = f64::ceil(width as f64 / cell_size) as u64;
        let grid_height = f64::ceil(height as f64 / cell_size) as u64;
        let grid = vec![None; (grid_width * grid_height) as usize];
        let queue = Vec::default();
        PoissonDiscSampler {
            round,
            width,
            height,
            radius,
            radius2,
            cell_size,
            grid_width,
            grid_height,
            max_samples,
            grid,
            queue,
            rng,
        }
    }
    #[tracing::instrument]
    pub fn step(&mut self) -> PoissonEvent {
        self.round = self.round.wrapping_add(1);
        // Round 0: Return the origin
        if self.round == 0 {
            // Pick the first Sample
            return PoissonEvent::Origin {
                point: self.sample(self.width as f64 / 2.0, self.height as f64 / 2.0),
            };
        }
        // Round N (Q: 0): Empty Queue, Return Complete
        if self.queue.is_empty() {
            return PoissonEvent::Complete;
        }
        // Round N (Q: N): Non-Empty Queue, Make a new Candidate

        // Shuffle our queue then "peek" at the last element.
        self.queue.shuffle(&mut self.rng);
        let parent = self.queue.last().map(Clone::clone).unwrap();
        let seed = self.rng.gen::<f64>();

        // Make a new candidate.
        for j in 0..self.max_samples {
            let a = TAU * (seed + j as f64 / self.max_samples as f64);
            let r = self.radius + 0.000001;
            let x = parent.x + r * f64::cos(a);
            let y = parent.y + r * f64::sin(a);

            // Accept candidates that are inside the allowed extent
            // and farther than 2 * radius to all existing samples.
            if (0.0 <= x && x < self.width as f64)
                && (0.0 <= y && y < self.height as f64)
                && self.far(x, y)
            {
                return PoissonEvent::Created {
                    parent,
                    point: self.sample(x, y),
                };
            }
        }

        // If none of k candidates were accepted, remove it from the queue.
        let _r = self.queue.pop();
        PoissonEvent::Closed { point: parent }
    }
    #[tracing::instrument]
    fn far(&mut self, x: f64, y: f64) -> bool {
        let i = f64::floor(x / self.cell_size) as u64;
        let j = f64::floor(y / self.cell_size) as u64;
        let i0 = i64::max(i as i64 - 2, 0) as u64;
        let j0 = i64::max(j as i64 - 2, 0) as u64;
        let i1 = u64::min(i + 3, self.grid_width as u64);
        let j1 = u64::min(j + 3, self.grid_height as u64);

        for j in j0..j1 {
            let o = j * self.grid_width;
            for i in i0..i1 {
                let idx = (o + i) as usize;
                if let Some(s) = &self.grid[idx] {
                    let dx = s.x - x;
                    let dy = s.y - y;
                    if dx * dx + dy * dy < self.radius2 {
                        return false;
                    }
                }
            }
        }
        true
    }
    #[tracing::instrument]
    fn sample(&mut self, x: f64, y: f64) -> PoissonPoint {
        let point = PoissonPoint {
            id: self.round,
            x,
            y,
        };
        let grid_x = f64::floor(x / self.cell_size) as u64;
        let grid_y = f64::floor(y / self.cell_size) as u64;
        let grid_idx = self.grid_width * grid_y + grid_x;
        self.grid[grid_idx as usize] = Some(point.clone());
        self.queue.push(point.clone());
        point
    }
}

const TAU: f64 = 2.0 * std::f64::consts::PI;

#[cfg(test)]
mod tests {
    use crate::point::{PoissonDiscSampler, PoissonEvent};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::collections::BTreeMap;
    use tracing::trace;
    use tracing_test::traced_test;

    #[test]
    #[traced_test]
    fn test_small() {
        run_test(100, 100, 10.0, 0);
    }

    fn run_test(width: u64, height: u64, radius: f64, seed: u64) {
        let mut sampler =
            PoissonDiscSampler::new(width, height, radius, SmallRng::seed_from_u64(seed));
        let mut nodes = BTreeMap::new();
        let mut edges = Vec::new();
        loop {
            match sampler.step() {
                PoissonEvent::Origin { point } => {
                    trace!("Origin Point: {}", &point);
                    nodes.insert(point.id, point);
                }
                PoissonEvent::Created { parent, point } => {
                    trace!("Created Point: {}", &point);
                    edges.push((parent.id, point.id));
                    nodes.insert(point.id, point);
                }
                PoissonEvent::Closed { point } => {
                    trace!("Closed Point: {}", &point);
                }
                PoissonEvent::Complete => {
                    trace!("Completed");
                    break;
                }
            }
        }
        println!("Poisson: Nodes({}), Edges({})", nodes.len(), edges.len());
        for (_id, node) in &nodes {
            println!("Point {}", node);
        }
        for (parent, child) in edges {
            let p = nodes.get(&parent).expect("Parent Node Missing").clone();
            let c = nodes.get(&child).expect("Child Node Missing").clone();
            println!("Edge[{}, {}]", p, c);
        }
    }
}
