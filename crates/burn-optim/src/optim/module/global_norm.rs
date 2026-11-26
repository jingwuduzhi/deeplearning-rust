use super::GradAdaptor;
use crate::{
    GradientsParams,
    grad_clipping::{apply_scale, into_norm_precision, norm_scale},
};
use alloc::vec::Vec;
use burn_core::{
    module::{Module, ModuleVisitor, Param},
    tensor::{DType, Device, Tensor},
};

struct NormCollector<'a> {
    source: &'a mut GradAdaptor,
    gradients: GradientsParams,
    device_sums: Vec<(Device, Tensor<1>)>,
}

impl ModuleVisitor for NormCollector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        // Removing by ID counts shared parameters once and resolves all sources in step_multi.
        let Some((grad, device)) = self.source.remove::<D>(param.id) else {
            return;
        };

        let sum = into_norm_precision(grad.clone()).square().sum();
        match self.device_sums.iter_mut().find(|(d, _)| *d == device) {
            Some((_, local)) => {
                let dtype = if sum.dtype() == DType::F64 {
                    DType::F64
                } else {
                    local.dtype()
                };
                *local = local.clone().cast(dtype) + sum.cast(dtype);
            }
            None => self.device_sums.push((device, sum)),
        }
        self.gradients.register(param.id, grad);
    }
}

pub(super) struct GlobalNorm {
    scales: Vec<(Device, Tensor<1>)>,
}

impl GlobalNorm {
    /// Replace the source with its accumulated gradients and return one copy of the scale per device.
    pub(super) fn prepare<M: Module>(module: &M, grads: &mut GradAdaptor, max_norm: f32) -> Self {
        let mut collector = NormCollector {
            source: grads,
            gradients: GradientsParams::new(),
            device_sums: Vec::new(),
        };
        module.visit(&mut collector);
        let NormCollector {
            gradients,
            device_sums,
            ..
        } = collector;
        *grads = gradients.into();

        Self {
            scales: Self::device_scales(device_sums, max_norm),
        }
    }

    fn device_scales(
        mut sums: Vec<(Device, Tensor<1>)>,
        max_norm: f32,
    ) -> Vec<(Device, Tensor<1>)> {
        if sums.is_empty() {
            return sums;
        }
        // Reduce on a device known to support F64 if any gradient uses it.
        let root = sums
            .iter()
            .position(|(_, s)| s.dtype() == DType::F64)
            .unwrap_or(0);
        sums.swap(0, root);
        let (device, first) = &sums[0];
        let dtype = first.dtype();
        let total = sums[1..].iter().fold(first.clone(), |total, (_, local)| {
            total + local.clone().to_device(device).cast(dtype)
        });
        let scale = norm_scale(total, max_norm);
        for (device, local) in &mut sums[1..] {
            *local = scale.clone().cast(local.dtype()).to_device(device);
        }
        sums[0].1 = scale;
        sums
    }

    pub(super) fn clip<const D: usize>(&self, grad: Tensor<D>, device: &Device) -> Tensor<D> {
        let (_, scale) = self
            .scales
            .iter()
            .find(|(d, _)| d == device)
            .expect("gradient device contributed a norm");
        let dtype = grad.dtype();
        apply_scale(grad, scale.clone(), dtype)
    }
}
