use burn_core as burn;

use burn::{
    config::Config,
    tensor::{DType, Tensor},
};

/// Gradient Clipping provides a way to mitigate exploding gradients
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum GradientClippingConfig {
    /// Clip the gradient by value.
    Value(f32),

    /// Limit the L2 norm of each gradient tensor independently.
    Norm(f32),

    /// Limit the joint L2 norm of all parameter gradients in a module optimizer step.
    GlobalNorm(f32),
}

impl GradientClippingConfig {
    /// Initialize the gradient clipping.
    ///
    /// # Returns
    ///
    /// The gradient clipping.
    pub fn init(&self) -> GradientClipping {
        match self {
            GradientClippingConfig::Value(val) => GradientClipping::Value(*val),
            GradientClippingConfig::Norm(val) => GradientClipping::Norm(*val),
            GradientClippingConfig::GlobalNorm(val) => GradientClipping::GlobalNorm(*val),
        }
    }
}

/// Gradient Clipping provides a way to mitigate exploding gradients
/// before an optimizer update.
#[derive(Clone)]
#[non_exhaustive]
pub enum GradientClipping {
    /// Clip the gradient by value.
    Value(f32),

    /// Limit the L2 norm of each gradient tensor independently.
    Norm(f32),

    /// Limit the joint L2 norm of all parameter gradients in a module optimizer step.
    GlobalNorm(f32),
}

impl GradientClipping {
    /// Clip a single gradient. For a single tensor, the global norm equals its own L2 norm.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to clip.
    ///
    /// # Returns
    ///
    /// The clipped gradient.
    pub fn clip_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        match self {
            GradientClipping::Value(threshold) => self.clip_by_value(grad, *threshold),
            GradientClipping::Norm(max_norm) | GradientClipping::GlobalNorm(max_norm) => {
                self.clip_by_norm(grad, *max_norm)
            }
        }
    }

    fn clip_by_value<const D: usize>(&self, grad: Tensor<D>, threshold: f32) -> Tensor<D> {
        let greater_mask = grad.clone().greater_scalar(threshold);
        let lower_mask = grad.clone().lower_scalar(-threshold);

        let clipped_grad = grad.mask_fill(greater_mask, threshold);

        clipped_grad.mask_fill(lower_mask, -threshold)
    }

    fn clip_by_norm<const D: usize>(&self, grad: Tensor<D>, threshold: f32) -> Tensor<D> {
        let dtype = grad.dtype();
        let grad = into_norm_precision(grad);
        let scale = norm_scale(grad.clone().square().sum(), threshold);
        apply_scale(grad, scale, dtype)
    }
}

/// Widen before squaring so FP16 gradients above 256 have finite squares.
pub(crate) fn into_norm_precision<const D: usize>(grad: Tensor<D>) -> Tensor<D> {
    match grad.dtype() {
        DType::F64 | DType::F32 => grad,
        _ => grad.dequantize().cast(DType::F32),
    }
}

pub(crate) fn norm_scale(sum_squares: Tensor<1>, max_norm: f32) -> Tensor<1> {
    let norm = sum_squares.sqrt();
    let within_limit = norm.clone().lower_equal_scalar(max_norm);
    (max_norm / norm).mask_fill(within_limit, 1.0)
}

pub(crate) fn apply_scale<const D: usize>(
    grad: Tensor<D>,
    scale: Tensor<1>,
    dtype: DType,
) -> Tensor<D> {
    let scaled = grad.dequantize().cast(scale.dtype()) * scale.unsqueeze();
    match dtype {
        DType::QFloat(scheme) => scaled.quantize_dynamic(&scheme),
        _ => scaled.cast(dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, Tensor};

    #[test]
    fn test_clip_by_value() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Value(0.5).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.iter::<f32>() {
            assert!(value <= 0.5);
        }
    }

    #[test]
    fn test_clip_by_norm() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Norm(2.2).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.iter::<f32>() {
            assert!(value <= 0.88);
        }
    }
    #[test]
    fn test_clip_by_norm_no_clipping() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [[0.3, 0.4, 0.5, 0.2], [0.1, 0.6, 0.3, 0.4]],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Norm(2.2).clip_gradient(gradient.clone());

        clipped_gradient
            .into_data()
            .assert_eq(&gradient.into_data(), true);
    }

    #[test]
    fn test_clip_by_norm_f16_does_not_overflow() {
        let gradient = Tensor::<1>::from_floats([300.0], &Default::default()).cast(FloatDType::F16);

        let clipped_gradient = GradientClipping::Norm(1.0).clip_gradient(gradient);

        assert_eq!(clipped_gradient.dtype(), DType::F16);
        let actual = clipped_gradient.into_scalar::<f32>();
        assert!((actual - 1.0).abs() < 1e-3, "actual={actual}");
    }

    #[test]
    fn test_clip_by_norm_f16_norm_above_f16_max() {
        let gradient =
            Tensor::<1>::from_floats([60000.0, 60000.0], &Default::default()).cast(FloatDType::F16);

        let clipped_gradient = GradientClipping::Norm(1.0).clip_gradient(gradient);

        let expected = core::f32::consts::FRAC_1_SQRT_2;
        for value in clipped_gradient.into_data().iter::<f32>() {
            assert!((value - expected).abs() < 1e-3, "value={value}");
        }
    }
}
