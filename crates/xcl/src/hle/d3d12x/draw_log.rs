//! `XCL_DRED=1`: which pipelines a command list drew / dispatched with.
//!
//! DRED's breadcrumbs name the list and the operation that was running when the GPU hung, not
//! the pipeline. Each console list (`bind::State`) therefore keeps the last `RING` draws,
//! dispatches and indirect executions – a fixed array, no allocation, cleared on `Reset` – and
//! `log_device_removed` prints the newest ones of the list DRED names (`pCommandList` is the
//! list object itself). Recorded only with `XCL_DRED`, so the hot path pays one relaxed load.

use std::fmt;

/// events kept per list; the oldest is overwritten
pub(super) const RING: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Kind {
    #[default]
    Draw,
    DrawIndexed,
    Dispatch,
    ExecuteIndirect,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Kind::Draw => "draw",
            Kind::DrawIndexed => "draw-indexed",
            Kind::Dispatch => "dispatch",
            Kind::ExecuteIndirect => "execute-indirect",
        })
    }
}

/// The console shaders (dump hashes) a pipeline was made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shaders {
    Graphics { vs: u64, ps: u64 },
    Compute { cs: u64 },
}

impl fmt::Display for Shaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shaders::Graphics { vs, ps } => write!(f, "vs {vs:016x} ps {ps:016x}"),
            Shaders::Compute { cs } => write!(f, "cs {cs:016x}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Event {
    pub kind: Kind,
    /// the pipeline the title bound (a background build's stand-in until it is resolved)
    pub pso: usize,
}

pub(super) struct Ring {
    events: [Event; RING],
    /// events pushed since the last `clear` (more than `RING` once it has wrapped)
    total: u64,
}

impl Default for Ring {
    fn default() -> Self {
        Self { events: [Event::default(); RING], total: 0 }
    }
}

impl Ring {
    pub fn push(&mut self, kind: Kind, pso: usize) {
        self.events[self.total as usize % RING] = Event { kind, pso };
        self.total += 1;
    }

    pub fn clear(&mut self) {
        self.total = 0;
    }

    #[cfg(test)]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The newest `n` events (at most those still kept), oldest first.
    pub fn recent(&self, n: usize) -> impl Iterator<Item = Event> + '_ {
        let take = (n.min(RING) as u64).min(self.total);
        (self.total - take..self.total).map(|i| self.events[i as usize % RING])
    }

    /// `"7 since Reset, last 2: dispatch cs … (pso 0x…) > draw vs … ps … (pso 0x…)"`, or None
    /// for a list that recorded nothing. `shaders` maps a pipeline to its console shaders.
    pub fn summary(&self, n: usize, shaders: impl Fn(usize) -> Option<Shaders>) -> Option<String> {
        if self.total == 0 {
            return None;
        }
        let last: Vec<String> = self
            .recent(n)
            .map(|e| match shaders(e.pso) {
                Some(s) => format!("{} {s} (pso {:#x})", e.kind, e.pso),
                None => format!("{} pso {:#x} (shaders unknown)", e.kind, e.pso),
            })
            .collect();
        Some(format!("{} since Reset, last {}: {}", self.total, last.len(), last.join(" > ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(pso: usize) -> Option<Shaders> {
        match pso {
            0x10 => Some(Shaders::Graphics { vs: 0xaa, ps: 0xbb }),
            0x20 => Some(Shaders::Compute { cs: 0xcc }),
            _ => None,
        }
    }

    #[test]
    fn an_empty_ring_has_no_summary() {
        assert_eq!(Ring::default().summary(4, lookup), None);
    }

    #[test]
    fn the_summary_names_the_newest_events_oldest_first() {
        let mut r = Ring::default();
        r.push(Kind::Draw, 0x10);
        r.push(Kind::Dispatch, 0x20);
        r.push(Kind::ExecuteIndirect, 0x30);
        assert_eq!(
            r.summary(2, lookup).unwrap(),
            "3 since Reset, last 2: dispatch cs 00000000000000cc (pso 0x20) > execute-indirect pso 0x30 (shaders unknown)"
        );
        assert_eq!(
            r.summary(8, lookup).unwrap(),
            "3 since Reset, last 3: draw vs 00000000000000aa ps 00000000000000bb (pso 0x10) > dispatch cs 00000000000000cc (pso 0x20) > execute-indirect pso 0x30 (shaders unknown)"
        );
    }

    #[test]
    fn the_ring_keeps_only_the_newest_events_once_full() {
        let mut r = Ring::default();
        for i in 0..RING * 2 + 5 {
            r.push(if i % 2 == 0 { Kind::Draw } else { Kind::DrawIndexed }, i);
        }
        assert_eq!(r.total(), (RING * 2 + 5) as u64);
        let all: Vec<usize> = r.recent(usize::MAX).map(|e| e.pso).collect();
        assert_eq!(all.len(), RING);
        assert_eq!(all, (RING + 5..RING * 2 + 5).collect::<Vec<_>>());
        let last: Vec<usize> = r.recent(3).map(|e| e.pso).collect();
        assert_eq!(last, [RING * 2 + 2, RING * 2 + 3, RING * 2 + 4]);
    }

    #[test]
    fn clear_forgets_the_recording() {
        let mut r = Ring::default();
        r.push(Kind::Dispatch, 0x20);
        r.clear();
        assert_eq!(r.recent(4).count(), 0);
        assert_eq!(r.summary(4, lookup), None);
        r.push(Kind::Draw, 0x10);
        assert_eq!(r.summary(4, lookup).unwrap(), "1 since Reset, last 1: draw vs 00000000000000aa ps 00000000000000bb (pso 0x10)");
    }
}
