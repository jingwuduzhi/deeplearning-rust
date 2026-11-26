use super::{L2Norm, scale_gradient};
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
    /// For quantized inputs, norm clipping reads the clipping decision to preserve the
    /// original encoding when the gradient is within the limit.
    ///
    /// # Panics
    ///
    /// Panics if a global norm limit is negative or NaN.
    pub fn clip_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        self.validate_global_limit();
        match self {
            GradientClipping::Value(threshold) => self.clip_by_value(grad, *threshold),
            GradientClipping::Norm(max_norm) | GradientClipping::GlobalNorm(max_norm) => {
                self.clip_by_norm(grad, *max_norm)
            }
        }
    }

    pub(crate) fn validate_global_limit(&self) {
        if let Self::GlobalNorm(limit) = self {
            assert!(
                *limit >= 0.,
                "GlobalNorm requires a nonnegative limit, got {limit}"
            );
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
        let sqrt_scale = L2Norm::of(grad.clone()).sqrt_scale(threshold);
        let keep_quantized = matches!(dtype, DType::QFloat(_))
            && threshold >= 0.
            && sqrt_scale.clone().into_scalar::<f64>() == 1.;
        let grad = if threshold < 0. {
            -super::norm_precision(grad)
        } else {
            grad
        };
        scale_gradient(grad, sqrt_scale, dtype, keep_quantized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, Tensor};

    #[test]
    fn norm_clips_tiny_finite_gradients() {
        let grad = Tensor::<1>::from_floats([3e-30, 4e-30], &Default::default());
        let scaled = GradientClipping::Norm(1e-30).clip_gradient(grad) * 1e30;
        let data = scaled.into_data();
        data.assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([0.6f32, 0.8]),
            Default::default(),
        );
    }

    #[test]
    fn norm_preserves_inactive_quantization() {
        use burn::tensor::{
            Device,
            quantization::{QuantValue, QuantizationParameters},
        };
        let device = Device::default();
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        let grad = Tensor::<1>::from_floats([1., 2., 0., -1.], &device).quantize(
            &scheme,
            QuantizationParameters {
                scales: Tensor::ones([1], &device),
                global: None,
            },
        );
        let expected = grad.clone().into_data();
        GradientClipping::GlobalNorm(10.)
            .clip_gradient(grad)
            .into_data()
            .assert_eq(&expected, true);
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires a nonnegative limit")]
    fn global_norm_rejects_negative_single_tensor_limit() {
        let grad = Tensor::<1>::ones([1], &Default::default());
        let _ = GradientClipping::GlobalNorm(-1.).clip_gradient(grad);
    }

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
