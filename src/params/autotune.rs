//! Auto-Tune ([`honeydew_audio::autotune`]), Honeydew's own effect: its params and value texts, in the engine's order
//! and laws.

use honeydew_audio::autotune::{
    self as a, DEFAULTS, FORMANT, HUMANIZE, KEY, MIX, PARAMS, RETUNE, SCALE,
};

use super::{Display, ModuleSpec, ParamSpec as P};

/// The `Key` menu, in the engine's order.
const KEYS: &[&str] = &[
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

static PARAM_SPECS: [P; PARAMS] = {
    let mut p = [P::ENABLE; PARAMS];
    p[KEY] = P::menu("Key", DEFAULTS[KEY], KEYS);
    p[SCALE] = P::menu("Scale", DEFAULTS[SCALE], a::SCALE_NAMES);
    p[RETUNE] = P::cont("Retune Speed", DEFAULTS[RETUNE], Display::Generic);
    p[HUMANIZE] = P::cont("Humanize", DEFAULTS[HUMANIZE], Display::Generic);
    // trunc(24v − 12) semitones, exactly as its text.
    p[FORMANT] = P::cont(
        "Formant",
        DEFAULTS[FORMANT],
        Display::Trunc64 {
            min: -12.0,
            max: 12.0,
            unit: " st",
        },
    );
    p[MIX] = P::cont("Mix", DEFAULTS[MIX], Display::Generic);
    p
};

/// The module. Not in [`super::modules`]: that is the song format's effects.
pub static MODULE: ModuleSpec = ModuleSpec::effect(
    a::AUTOTUNE,
    "Auto-Tune",
    "Auto-Tune",
    false,
    &PARAM_SPECS,
    &[],
);
