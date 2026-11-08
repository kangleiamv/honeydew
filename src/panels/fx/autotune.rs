//! Auto-Tune (`honeydew_audio::autotune`), UIHeight 7 U: Honeydew's own panel. Key and Scale pick the notes to snap
//! to, then Retune Speed, Humanize, Formant and Mix.

use honeydew_audio::autotune::{FORMANT, HUMANIZE, KEY, MIX, RETUNE, SCALE};

use crate::panels::build::*;

pub(in crate::panels) fn panel() -> Panel {
    Panel::new(hstack([
        spinner(1, "Key").p(KEY),
        spinner(2, "Scale").p(SCALE),
        knob(3, "Retune").p(RETUNE),
        knob(4, "Humanize").p(HUMANIZE),
        knob(5, "Formant").p(FORMANT),
        knob(6, "Mix").p(MIX),
    ]))
}
