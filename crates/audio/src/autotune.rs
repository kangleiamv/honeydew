//! Auto-Tune: an Antares-style real-time pitch corrector, a rack effect of its own.
//!
//! One monophonic pitch is tracked from the input's mono mix (the Pitcher's autocorrelation [`Tracker`]) and the
//! signal is resynthesised by the Pitcher's centred two-tap granular shifter (`pitcher::shifted`), so the module runs
//! on the audio thread without allocating. The controls are the familiar Auto-Tune ones: a `Key` and a `Scale` (the
//! pitch classes a note may snap to), `Retune Speed` (how fast the correction glides in), `Humanize` (the slow,
//! pseudorandom detune that keeps the result from sounding locked), `Formant` (the crude spectral compensation the
//! Pitcher's F-Shift uses) and `Mix`.
//!
//! Unlike the Pitcher, which takes its note mask from the channel strip's `CSTM`, this module's key and scale are
//! ordinary parameters: the mask is derived in [`AutoTuneState::refresh`], off the audio thread.

use std::f32::consts::TAU;

use crate::pitcher::{Search, Tracker, latency_for, read, shifted};

/// The module's type id.
pub const AUTOTUNE: u16 = 1009;
/// Its parameter count.
pub const PARAMS: usize = 7;

pub const ENABLE: usize = 0;
pub const KEY: usize = 1;
pub const SCALE: usize = 2;
pub const RETUNE: usize = 3;
pub const HUMANIZE: usize = 4;
pub const FORMANT: usize = 5;
pub const MIX: usize = 6;

/// A new module: on, C major, a moderate retune speed, no humanize or formant shift, fully wet.
pub const DEFAULTS: [f32; PARAMS] = [1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 1.0];

/// The `Scale` menu, in the registry's order. The audio crate owns it so the panel and the engine cannot disagree.
pub const SCALE_NAMES: &[&str] = &[
    "Major",
    "Minor",
    "Dorian",
    "Phrygian",
    "Lydian",
    "Mixolydian",
    "Chromatic",
];

/// Each scale's tones, C...B relative to the key, most significant bit C (1 = in scale), as `src/song/scales.rs`.
const SCALE_MASKS: [u16; 7] = [
    0b101011010101, // Major
    0b101101011010, // Minor
    0b101101010110, // Dorian
    0b110101011010, // Phrygian
    0b101010110101, // Lydian
    0b101011010110, // Mixolydian
    0b111111111111, // Chromatic
];

/// The lowest and highest pitch tracked, Hz (the Pitcher's bounds), and the correlation below which input is unvoiced.
const LOWEST: f32 = 55.0;
const HIGHEST: f32 = 1500.0;
const THRESHOLD: f32 = 0.3;
/// The farthest a detected note may be moved, in semitones: farther reads as an octave error and is left alone.
const MAX_SEMITONES: f32 = 2.0;
/// The most `Humanize` detunes the target, in cents.
const HUMANIZE_CENTS: f32 = 15.0;
/// Analyses between a new humanize target (each is 64 samples).
const HUMANIZE_PERIOD: u32 = 128;
/// The first humanize seed.
const HUMANIZE_SEED: u32 = 0x1234_5678;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoTuneError {
    InvalidValue,
    InvalidSampleRate,
    MismatchedBuffers,
}

pub struct AutoTuneState {
    params: [f32; PARAMS],
    sample_rate: f32,
    latency: usize,
    history: [Vec<f32>; 2],
    formant_history: [Vec<f32>; 2],
    write_index: usize,
    tracker: Tracker,
    /// The samples since the last pitch analysis, kept under 64.
    frames: usize,
    tracked_frequency: f32,
    voiced: bool,
    /// The correction ratio of the last analysis, held through unvoiced input.
    target_ratio: f32,
    ratio: f32,
    phase: f32,
    formant_coefficient: f32,
    formant_input_low: [f32; 2],
    formant_output_low: [f32; 2],
    /// The pitch classes (0 = C) a note may snap to, from `Key` and `Scale`.
    notes: [bool; 12],
    humanize_seed: u32,
    humanize_target: f32,
    humanize_cents: f32,
    humanize_clock: u32,
}

impl AutoTuneState {
    pub fn new(params: &[f32], sample_rate: f32) -> Result<Self, AutoTuneError> {
        validate(params, sample_rate)?;
        let latency = latency_for(sample_rate);
        let mut state = Self {
            params: DEFAULTS,
            sample_rate,
            latency,
            history: std::array::from_fn(|_| vec![0.0; latency * 3 + 4]),
            formant_history: std::array::from_fn(|_| vec![0.0; latency * 3 + 4]),
            write_index: 0,
            tracker: Tracker::new(sample_rate, LOWEST),
            frames: 0,
            tracked_frequency: 0.0,
            voiced: false,
            target_ratio: 1.0,
            ratio: 1.0,
            phase: 0.0,
            formant_coefficient: 0.0,
            formant_input_low: [0.0; 2],
            formant_output_low: [0.0; 2],
            notes: [true; 12],
            humanize_seed: HUMANIZE_SEED,
            humanize_target: 0.0,
            humanize_cents: 0.0,
            humanize_clock: 0,
        };
        state.refresh(params, sample_rate)?;
        Ok(state)
    }

    pub fn parameters(&self) -> &[f32; PARAMS] {
        &self.params
    }

    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    pub fn tracked_frequency(&self) -> f32 {
        self.tracked_frequency
    }

    /// The pitch classes (0 = C) a note may snap to, from `Key` and `Scale`.
    pub fn notes(&self) -> [bool; 12] {
        self.notes
    }

    /// The formant ratio the `Formant` control asks for (0.5...2.0; 1 is none).
    pub fn formant_ratio(&self) -> f32 {
        (self.formant_semitones() / 12.0).exp2()
    }

    pub fn refresh(&mut self, params: &[f32], sample_rate: f32) -> Result<(), AutoTuneError> {
        validate(params, sample_rate)?;
        if self.sample_rate != sample_rate {
            self.sample_rate = sample_rate;
            self.latency = latency_for(sample_rate);
            let length = self.latency * 3 + 4;
            for channel in 0..2 {
                self.history[channel].resize(length, 0.0);
                self.formant_history[channel].resize(length, 0.0);
            }
            self.tracker = Tracker::new(sample_rate, LOWEST);
            self.reset();
        }
        self.params.copy_from_slice(params);
        self.notes = notes_for(params[KEY], params[SCALE]);
        let cutoff = (500.0 * self.formant_ratio()).min(0.45 * sample_rate);
        self.formant_coefficient = (1.0 - (-TAU * cutoff / sample_rate).exp()).clamp(0.0, 1.0);
        Ok(())
    }

    /// The correction ratio that moves a pitch of `frequency` Hz to its nearest enabled tone, with the humanize
    /// detune added; 1 when there is no tone within [`MAX_SEMITONES`].
    pub fn target_for(&self, frequency: f32) -> f32 {
        if !frequency.is_finite() || frequency <= 0.0 {
            return 1.0;
        }
        let midi = 69.0 + 12.0 * (frequency / 440.0).log2();
        let Some(note) = self.nearest_note(midi) else {
            return 1.0;
        };
        let cents = self.humanize_cents * self.params[HUMANIZE];
        let semitones = note - midi + cents * 0.01;
        (semitones / 12.0).exp2().clamp(0.5, 2.0)
    }

    /// The nearest enabled note to `midi`, within [`MAX_SEMITONES`].
    fn nearest_note(&self, midi: f32) -> Option<f32> {
        let center = midi.round() as i32;
        (-2..=2)
            .map(|offset| center + offset)
            .filter(|note| self.notes[note.rem_euclid(12) as usize])
            .map(|note| (note, (note as f32 - midi).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|&(_, distance)| distance <= MAX_SEMITONES)
            .map(|(note, _)| note as f32)
    }

    /// `Formant` in whole semitones, `trunc(24v − 12)`, exactly as its text.
    fn formant_semitones(&self) -> f32 {
        (-12.0 + 24.0 * f64::from(self.params[FORMANT])) as i32 as f32
    }

    /// The ratio smoothing's time constant, seconds: 1 ms at Retune Speed 1, 0.3 s at 0.
    fn retune_seconds(&self) -> f32 {
        let speed = self.params[RETUNE].clamp(0.0, 1.0);
        0.001 + 0.3 * (1.0 - speed).powi(3)
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) -> Result<(), AutoTuneError> {
        if left.len() != right.len() {
            return Err(AutoTuneError::MismatchedBuffers);
        }
        for (l, r) in left.iter_mut().zip(right) {
            (*l, *r) = self.process_frame(*l, *r);
        }
        Ok(())
    }

    pub fn process_frame(&mut self, left: f32, right: f32) -> (f32, f32) {
        self.observe_pitch(0.5 * (left + right));
        let seconds = self.retune_seconds();
        let coefficient = (1.0 - (-1.0 / (seconds * self.sample_rate)).exp()).clamp(0.0, 1.0);
        self.ratio += coefficient * (self.target_ratio.clamp(0.5, 2.0) - self.ratio);
        let window = self.latency as f32;
        let phase = self.phase + (1.0 - self.ratio) / window;
        self.phase = phase - phase.floor();
        let formant_coefficient = self.formant_coefficient;
        let formant = self.formant_semitones() != 0.0;
        let mut output = [0.0; 2];
        for (channel, input) in [left, right].into_iter().enumerate() {
            self.formant_input_low[channel] +=
                formant_coefficient * (input - self.formant_input_low[channel]);
            self.history[channel][self.write_index] = input;
            self.formant_history[channel][self.write_index] = self.formant_input_low[channel];
            let dry = read(&self.history[channel], self.write_index, window);
            let shifted = shifted(&self.history[channel], self.write_index, self.phase, window);
            let wet = if formant {
                self.formant_output_low[channel] +=
                    formant_coefficient * (shifted - self.formant_output_low[channel]);
                let original_low = read(&self.formant_history[channel], self.write_index, window);
                shifted + 0.4 * (original_low - self.formant_output_low[channel])
            } else {
                shifted
            };
            output[channel] = if self.params[ENABLE] < 0.5 {
                dry
            } else {
                dry + self.params[MIX] * (wet - dry)
            };
        }
        self.write_index = (self.write_index + 1) % self.history[0].len();
        // Only its phase within 64 matters (`observe_pitch`): kept inside, so wasm32's usize never overflows.
        self.frames = (self.frames + 1) % 64;
        (output[0], output[1])
    }

    fn observe_pitch(&mut self, sample: f32) {
        self.tracker.push(sample);
        if !self.frames.is_multiple_of(64) {
            return;
        }
        match self.tracker.search(HIGHEST, LOWEST, THRESHOLD) {
            Search::Waiting => {}
            Search::Pitch(hz) => {
                self.tracked_frequency = hz;
                self.voiced = true;
            }
            Search::Unvoiced | Search::Edge => self.voiced = false,
        }
        self.advance_humanize();
        if self.voiced {
            self.target_ratio = self.target_for(self.tracked_frequency);
        }
    }

    /// The slow pseudorandom target `Humanize` adds, smoothed between its updates.
    fn advance_humanize(&mut self) {
        self.humanize_clock = (self.humanize_clock + 1) % HUMANIZE_PERIOD;
        if self.humanize_clock == 0 {
            self.humanize_seed = self
                .humanize_seed
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let unit = (self.humanize_seed >> 8) as f32 / 16_777_216.0;
            self.humanize_target = (2.0 * unit - 1.0) * HUMANIZE_CENTS;
        }
        self.humanize_cents += 0.02 * (self.humanize_target - self.humanize_cents);
    }

    pub fn reset(&mut self) {
        for channel in 0..2 {
            self.history[channel].fill(0.0);
            self.formant_history[channel].fill(0.0);
        }
        self.tracker.reset();
        self.write_index = 0;
        self.frames = 0;
        self.tracked_frequency = 0.0;
        self.voiced = false;
        self.target_ratio = 1.0;
        self.ratio = 1.0;
        self.phase = 0.0;
        self.formant_input_low = [0.0; 2];
        self.formant_output_low = [0.0; 2];
        self.humanize_seed = HUMANIZE_SEED;
        self.humanize_target = 0.0;
        self.humanize_cents = 0.0;
        self.humanize_clock = 0;
    }
}

/// The pitch classes (0 = C) a note in `key` (normalised, C = 0) on `scale` (normalised) may snap to.
pub fn notes_for(key: f32, scale: f32) -> [bool; 12] {
    let key = item_index(key, 12);
    let mask = SCALE_MASKS[item_index(scale, SCALE_MASKS.len())];
    let mut notes = [false; 12];
    for bit in 0..12 {
        if mask & (1 << bit) != 0 {
            // Bit 11 is the scale's C, bit 0 its B.
            notes[(key + 11 - bit) % 12] = true;
        }
    }
    notes
}

/// A menu's item for normalised `v`: `round(v·(len − 1))`, halves away from zero (the registry's `menu_index`).
fn item_index(v: f32, len: usize) -> usize {
    if len < 2 {
        return 0;
    }
    (v.clamp(0.0, 1.0) * (len - 1) as f32)
        .round()
        .min((len - 1) as f32) as usize
}

fn validate(params: &[f32], sample_rate: f32) -> Result<(), AutoTuneError> {
    if !sample_rate.is_finite() || sample_rate < 2000.0 {
        return Err(AutoTuneError::InvalidSampleRate);
    }
    if params.len() != PARAMS
        || params
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(AutoTuneError::InvalidValue);
    }
    Ok(())
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
    /// parabolically: the 300...700 Hz range the tests' sines lie in.
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

    fn state(rate: f32) -> AutoTuneState {
        AutoTuneState::new(&DEFAULTS, rate).unwrap()
    }

    fn render(mut state: AutoTuneState, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&sample| state.process_frame(sample, sample).0)
            .collect()
    }

    #[test]
    fn a_sharp_note_is_pulled_onto_a_major_scale() {
        // +0.4 semitones sharp of A4 (≈450.3 Hz), on A major: the nearest tone is A4, and Chromatic is not needed.
        let mut params = DEFAULTS;
        (params[KEY], params[SCALE], params[RETUNE]) = (9.0 / 11.0, 0.0, 1.0);
        let state = AutoTuneState::new(&params, RATE).unwrap();
        let input = sine(440.0 * (0.4f32 / 12.0).exp2(), 8_000);
        let output = render(state, &input);
        let got = frequency(&output, 4_000, 7_600);
        assert!((got - 440.0).abs() < 4.0, "{got} Hz");
    }

    #[test]
    fn an_in_tune_note_keeps_its_pitch() {
        let mut params = DEFAULTS;
        (params[KEY], params[SCALE], params[RETUNE]) = (9.0 / 11.0, 0.0, 1.0);
        let state = AutoTuneState::new(&params, RATE).unwrap();
        let output = render(state, &sine(440.0, 8_000));
        let got = frequency(&output, 4_000, 7_600);
        assert!((got - 440.0).abs() < 2.0, "{got} Hz");
    }

    #[test]
    fn the_key_and_scale_pick_the_notes() {
        // C major: C D E F G A B, no sharps.
        let major = notes_for(0.0, 0.0);
        for (note, on) in major.iter().enumerate() {
            assert_eq!(
                *on,
                [0, 2, 4, 5, 7, 9, 11].contains(&note),
                "C major {note}"
            );
        }
        // A minor: A B C D E F G.
        let minor = notes_for(9.0 / 11.0, 1.0 / 6.0);
        for (note, on) in minor.iter().enumerate() {
            assert_eq!(
                *on,
                [9, 11, 0, 2, 4, 5, 7].contains(&note),
                "A minor {note}"
            );
        }
        // Chromatic enables everything, whatever the key.
        assert_eq!(notes_for(3.0 / 11.0, 1.0), [true; 12]);
        // The state derives the same mask from its params.
        let mut params = DEFAULTS;
        (params[KEY], params[SCALE]) = (2.0 / 11.0, 1.0 / 6.0);
        assert_eq!(
            AutoTuneState::new(&params, RATE).unwrap().notes(),
            notes_for(2.0 / 11.0, 1.0 / 6.0)
        );
    }

    #[test]
    fn the_farthest_note_moved_is_two_semitones() {
        let mut state = state(RATE);
        // Only C: the pitch classes are sparse, so the distance to the nearest tone can pass the limit.
        state.notes = [false; 12];
        state.notes[0] = true;
        // Midway between A and A# (midi 69.5): C5 is 2.5 semitones up, past the limit, so it is left alone.
        let halfway = 440.0 * (0.5f32 / 12.0).exp2();
        assert_eq!(state.target_for(halfway), 1.0);
        // Midway between A# and B (midi 70.5): C5 is 1.5 up and within range.
        let nearer = 440.0 * (1.5f32 / 12.0).exp2();
        let ratio = state.target_for(nearer);
        assert!((ratio - (1.5f32 / 12.0).exp2()).abs() < 1e-4, "{ratio}");
        // No pitch stays untouched.
        assert_eq!(state.target_for(0.0), 1.0);
    }

    #[test]
    fn bypass_and_a_dry_mix_leave_the_signal_alone() {
        let input = sine(449.0, 2_000);
        let mut off = DEFAULTS;
        off[ENABLE] = 0.0;
        let mut dry = DEFAULTS;
        dry[MIX] = 0.0;
        let a = render(AutoTuneState::new(&off, RATE).unwrap(), &input);
        let b = render(AutoTuneState::new(&dry, RATE).unwrap(), &input);
        assert_eq!(a, b);
        // A wet pass differs from the dry one on a detuned note.
        let mut wet = DEFAULTS;
        (wet[KEY], wet[SCALE], wet[RETUNE]) = (9.0 / 11.0, 0.0, 1.0);
        let c = render(AutoTuneState::new(&wet, RATE).unwrap(), &input);
        assert_ne!(a, c);
    }

    #[test]
    fn formant_shift_is_none_at_the_middle() {
        let mut state = state(RATE);
        assert!((state.formant_ratio() - 1.0).abs() < 1e-6);
        state.params[FORMANT] = 0.0;
        assert!((state.formant_ratio() - 0.5).abs() < 1e-6);
        state.params[FORMANT] = 1.0;
        assert!((state.formant_ratio() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_settings_are_rejected() {
        assert_eq!(
            AutoTuneState::new(&DEFAULTS, 1_000.0).err(),
            Some(AutoTuneError::InvalidSampleRate)
        );
        let mut params = DEFAULTS;
        params[RETUNE] = 1.5;
        assert_eq!(
            AutoTuneState::new(&params, RATE).err(),
            Some(AutoTuneError::InvalidValue)
        );
        let mut state = state(RATE);
        assert_eq!(
            state.process(&mut [0.0; 2], &mut [0.0; 3]),
            Err(AutoTuneError::MismatchedBuffers)
        );
    }

    #[test]
    fn the_latency_is_the_pitchers_window() {
        let state = state(RATE);
        assert_eq!(state.latency_samples(), latency_for(RATE));
        assert!(state.latency_samples() >= 4);
    }
}
