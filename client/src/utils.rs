#[inline(always)]
pub fn calc_scaling_factor(original: usize, target: f32) -> f32 {
    let original = original as f32;
    target / original
}
