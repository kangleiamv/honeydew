//! Offline pitch correction for the audio clip editor's `Correct Pitch` region op: a PitchNet-like snap of a
//! selection's pitch to the project scale, keeping its timing. Detection reuses the Pitcher's autocorrelation
//! [`Tracker`] (`crate::pitcher`); each voiced frame is moved to the nearest enabled semitone and the whole selection
//! is resynthesised by a centred two-tap granular shifter (the Pitcher's `shifted`, read around each output frame so
//! the delay the read costs is not added to the sample).
//!
//! Pure and app thread (it allocates), like [`crate::sample_edit`]. The ops take one `Vec<f32>` per channel (mono one,
//! stereo two, equal lengths) and a frame range `[s, e)`; frames past the end of a range are ignored, so the sample is
//! treated as followed by silence.

use std::ops::Range;

use crate::pitcher::{Search, Tracker};

/// The lowest pitch corrected, Hz (the Pitcher's lowest).
const LOWEST: f32 = 55.0;
/// The highest pitch corrected, Hz.
const HIGHEST: f32 = 1500.0;
/// Analysis frames a second.
const FRAMES_PER_SECOND: f32 = 100.0;
/// The tracker's normalized autocorrelation below which a frame is unvoiced (the Pitcher's threshold).
const THRESHOLD: f32 = 0.3;
/// The most a frame is moved, in semitones: a farther nearest tone reads as a detection octave error and is left alone.
const MAX_SEMITONES: f32 = 2.0;

/// `Correct Pitch`: snap the frames of `ch` in `r` to the nearest semitone [`notes`] enables (pitch classes, 0 = C),
/// by `amount` of the way (0...1; 1 fully to the note). A voiced frame whose nearest enabled tone is more than
/// [`MAX_SEMITONES`] away is left (a likely octave error). No channels, an empty range, or no enabled note: unchanged.
pub fn correct_pitch(
    ch: &mut [Vec<f32>],
    sample_rate: f32,
    r: Range<usize>,
    notes: [bool; 12],
    amount: f32,
) {
    let len = ch.first().map_or(0, Vec::len);
    let (start, end) = (r.start.min(len), r.end.min(len));
    if start >= end
        || !sample_rate.is_finite()
        || sample_rate < 2000.0
        || !notes.iter().any(|&on| on)
    {
        return;
    }
    let ratios = ratios(&mix(ch), sample_rate, start, end, notes, amount);
    if ratios.is_empty() {
        return;
    }
    let hop = hop_frames(sample_rate);
    let window = window_frames(sample_rate);
    let fade = window.min((end - start) / 2).max(1);
    for samples in ch.iter_mut().filter(|s| s.len() >= len) {
        let input = samples.clone();
        shift(samples, &input, &ratios, hop, start, end, window, fade);
    }
}

/// The tracker's hop in frames: [`FRAMES_PER_SECOND`] of the input.
fn hop_frames(sample_rate: f32) -> usize {
    ((sample_rate / FRAMES_PER_SECOND).round() as usize).max(1)
}

/// The granular window in frames, at which the read taps span the two periods of [`LOWEST`] (the Pitcher's
/// `latency_for`): a multiple of four.
fn window_frames(sample_rate: f32) -> usize {
    ((sample_rate / LOWEST) as usize & !3).max(4)
}

/// The mono mix the pitch is detected on.
fn mix(ch: &[Vec<f32>]) -> Vec<f32> {
    let len = ch.first().map_or(0, Vec::len);
    let scale = 1.0 / ch.len().max(1) as f32;
    (0..len)
        .map(|i| {
            ch.iter()
                .map(|c| c.get(i).copied().unwrap_or(0.0))
                .sum::<f32>()
                * scale
        })
        .collect()
}

/// One correction ratio per analysis frame of `[start, end)`: the frame at `start + k·hop` holds the ratio the
/// tracker read there ([`snap`]), and every sample between two frames reads the linear interpolation (the ends hold).
fn ratios(
    mono: &[f32],
    sample_rate: f32,
    start: usize,
    end: usize,
    notes: [bool; 12],
    amount: f32,
) -> Vec<f32> {
    let hop = hop_frames(sample_rate);
    let positions: Vec<usize> = std::iter::successors(Some(start), |&p| Some(p + hop))
        .take_while(|&p| p < end)
        .collect();
    let mut tracker = Tracker::new(sample_rate, LOWEST);
    let mut out = Vec::with_capacity(positions.len());
    let mut next = 0;
    for (i, &sample) in mono.iter().enumerate().take(end) {
        tracker.push(sample);
        if next < positions.len() && i == positions[next] {
            out.push(snap(&mut tracker, notes, amount));
            next += 1;
        }
    }
    // Frames the input never reached (a range past its end) hold as in tune.
    out.resize(positions.len(), 1.0);
    smooth(&mut out);
    out
}

/// The ratio for the frequency the tracker reads now: 1 unvoiced or beyond [`MAX_SEMITONES`], else [`ratio_for`].
fn snap(tracker: &mut Tracker, notes: [bool; 12], amount: f32) -> f32 {
    match tracker.search(HIGHEST, LOWEST, THRESHOLD) {
        Search::Pitch(freq) => ratio_for(freq, &notes, amount),
        Search::Waiting | Search::Unvoiced | Search::Edge => 1.0,
    }
}

/// The ratio that moves a pitch of `freq` Hz `amount` of the way to the nearest semitone `notes` enables (0 = C),
/// clamped to an octave. 1 when none is within [`MAX_SEMITONES`].
fn ratio_for(freq: f32, notes: &[bool; 12], amount: f32) -> f32 {
    if !freq.is_finite() || freq <= 0.0 {
        return 1.0;
    }
    let midi = 69.0 + 12.0 * (freq / 440.0).log2();
    let base = midi.round() as i32;
    let mut target = None;
    let mut nearest = MAX_SEMITONES + 1.0;
    for note in base - 2..=base + 2 {
        if notes[note.rem_euclid(12) as usize] {
            let distance = (note as f32 - midi).abs();
            if distance < nearest {
                nearest = distance;
                target = Some(note as f32);
            }
        }
    }
    let Some(target) = target.filter(|_| nearest <= MAX_SEMITONES) else {
        return 1.0;
    };
    let semitones = amount.clamp(0.0, 1.0) * (target - midi);
    (semitones / 12.0).exp2().clamp(0.5, 2.0)
}

/// A frame's ratio as the average of it and its neighbours: a lone octave slip in one frame cannot jerk the read.
fn smooth(ratios: &mut [f32]) {
    if ratios.len() < 3 {
        return;
    }
    let source = ratios.to_vec();
    for (k, value) in (ratios.iter_mut().enumerate())
        .take(source.len() - 1)
        .skip(1)
    {
        *value = 0.25 * source[k - 1] + 0.5 * source[k] + 0.25 * source[k + 1];
    }
}

/// The ratio at sample `i` of `[start, end)`, the linear interpolation of the [`ratios`] at its frames.
fn ratio_at(ratios: &[f32], start: usize, hop: usize, i: usize) -> f32 {
    let position = (i - start) as f32 / hop as f32;
    let frame = position as usize;
    match (ratios.get(frame), ratios.get(frame + 1)) {
        (Some(&a), Some(&b)) => a + (b - a) * (position - frame as f32),
        (Some(&last), None) => last,
        _ => 1.0,
    }
}

/// Resynthesise `out[start..end]` from `input` at the [`ratio_at`] rates: the input around each frame is read by a
/// centred two-tap granular tap ([`granular`]) and crossfaded with the dry frames over `fade` at each end.
#[allow(clippy::too_many_arguments)]
fn shift(
    out: &mut [f32],
    input: &[f32],
    ratios: &[f32],
    hop: usize,
    start: usize,
    end: usize,
    window: usize,
    fade: usize,
) {
    let w = window as f32;
    let mut phase = 0.0f32;
    for (offset, sample) in out[start..end].iter_mut().enumerate() {
        let i = start + offset;
        let ratio = ratio_at(ratios, start, hop, i);
        phase += (1.0 - ratio) / w;
        phase -= phase.floor();
        let wet = granular(input, i + window, phase, w);
        let dry = input.get(i).copied().unwrap_or(0.0);
        let edge = (i - start).min(end - 1 - i) as f32 / fade as f32;
        *sample = dry + (wet - dry) * edge.min(1.0);
    }
}

/// The two-tap granular read the Pitcher uses, around the frame at `write_index`: at `phase` 0 and 0.5 the tap lands
/// exactly on it (`write_index - window`), so a constant 1 ratio is the input untouched.
fn granular(input: &[f32], write_index: usize, phase: f32, window: f32) -> f32 {
    let other = (phase + 0.5).fract();
    let weight = 1.0 - (2.0 * phase - 1.0).abs();
    let other_weight = 1.0 - (2.0 * other - 1.0).abs();
    let base = 0.5 * window;
    let at = write_index as f32;
    weight * read(input, at - (base + phase * window))
        + other_weight * read(input, at - (base + other * window))
}

/// `input` at the fractional `position`, linearly interpolated; 0 before its start and held at its last frame past it.
fn read(input: &[f32], position: f32) -> f32 {
    if position <= 0.0 || input.is_empty() {
        return 0.0;
    }
    let last = input.len() - 1;
    let index = (position as usize).min(last);
    let next = (index + 1).min(last);
    input[index] + (input[next] - input[index]) * (position - index as f32)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    const RATE: f32 = 8_000.0;

    fn sine(freq: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (TAU * freq * i as f32 / RATE).sin() * 0.5)
            .collect()
    }

    /// The fundamental of `samples` in [from, to), by the autocorrelation lag with the highest score, refined
    /// parabolically: the 300...700 Hz range a test sine lies in.
    fn frequency(samples: &[f32], from: usize, to: usize) -> f32 {
        let x = &samples[from..to];
        let corr = |lag: usize| -> f32 { (0..x.len() - lag).map(|i| x[i] * x[i + lag]).sum() };
        let (min, max) = ((RATE / 700.0) as usize, (RATE / 300.0) as usize);
        let scores: Vec<f32> = (min..=max).map(corr).collect();
        let best = (0..scores.len())
            .max_by(|a, b| scores[*a].total_cmp(&scores[*b]))
            .unwrap();
        let lag = min + best;
        let current = scores[best];
        let (below, above) = if best == 0 || best + 1 == scores.len() {
            (current, current)
        } else {
            (scores[best - 1], scores[best + 1])
        };
        let denominator = below - 2.0 * current + above;
        let offset = if denominator == 0.0 {
            0.0
        } else {
            0.5 * (below - above) / denominator
        };
        RATE / (lag as f32 + offset.clamp(-0.5, 0.5))
    }

    #[test]
    fn a_flat_sine_snaps_to_the_nearest_semitone() {
        // +0.4 semitones sharp of A4 (≈450.3 Hz): snapped down to 440.
        let mut ch = vec![sine(440.0 * (0.4f32 / 12.0).exp2(), 8_000)];
        correct_pitch(&mut ch, RATE, 800..7_200, [true; 12], 1.0);
        let got = frequency(&ch[0], 1_600, 6_400);
        assert!((got - 440.0).abs() < 3.0, "{got} Hz");
    }

    #[test]
    fn an_in_tune_sine_keeps_its_pitch() {
        let mut ch = vec![sine(440.0, 8_000)];
        correct_pitch(&mut ch, RATE, 800..7_200, [true; 12], 1.0);
        let got = frequency(&ch[0], 1_600, 6_400);
        assert!((got - 440.0).abs() < 1.0, "{got} Hz");
    }

    #[test]
    fn silence_and_a_disabled_scale_are_untouched() {
        let quiet = vec![vec![0.0f32; 8_000]];
        let mut ch = quiet.clone();
        correct_pitch(&mut ch, RATE, 0..8_000, [true; 12], 1.0);
        assert_eq!(ch, quiet);
        let mut ch = vec![sine(449.0, 8_000)];
        let before = ch.clone();
        correct_pitch(&mut ch, RATE, 800..7_200, [false; 12], 1.0);
        assert_eq!(ch, before);
    }

    #[test]
    fn output_keeps_the_samples_length_and_the_untouched_frames() {
        let mut ch = vec![sine(449.0, 8_000)];
        let before = ch[0].clone();
        correct_pitch(&mut ch, RATE, 2_000..4_000, [true; 12], 1.0);
        assert_eq!(ch[0].len(), before.len());
        // The crossfade holds the range's own edges, and everything outside it, as it was.
        assert_eq!(ch[0][..2_000], before[..2_000]);
        assert_eq!(ch[0][4_000..], before[4_000..]);
    }

    #[test]
    fn the_ratio_reaches_the_nearest_enabled_note() {
        // Only A# (10) on: A4 (midi 69) is pulled up a semitone.
        let mut sharp = [false; 12];
        sharp[10] = true;
        let ratio = ratio_for(440.0, &sharp, 1.0);
        assert!((ratio - (1.0f32 / 12.0).exp2()).abs() < 1e-4, "{ratio}");
        // Only A on, +0.4 semitones sharp: pulled down, but not all the way below A.
        let mut a = [false; 12];
        a[9] = true;
        let ratio = ratio_for(440.0 * (0.4f32 / 12.0).exp2(), &a, 1.0);
        assert!(ratio < 1.0 && ratio > 0.9, "{ratio}");
        // Half the way: the geometric middle.
        let half = ratio_for(440.0, &sharp, 0.5);
        assert!((half - (0.5f32 / 12.0).exp2()).abs() < 1e-4, "{half}");
        // A# at any octave stays A# (it is already a tone; pitch classes recur every octave).
        assert!((ratio_for(233.08, &sharp, 1.0) - 1.0).abs() < 1e-3);
    }
}
