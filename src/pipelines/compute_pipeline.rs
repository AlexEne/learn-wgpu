use crate::shader_compiler;
use wgpu::{self, BindGroupLayoutDescriptor};

pub struct ComputePipeline {
    compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    pub fn new(device: &wgpu::Device) -> ComputePipeline {
        let compute_shader_module = shader_compiler::compile_compute_shader(
            device,
            shader_compiler::ComputeShaderInput {
                shader_code: include_str!("../shaders/compute_prepass.comp"),
                file_name: "vertex_shader.vert",
                entry_point: "main",
            },
        );

        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Compute bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Compute Prepass Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Render Compute Prepass"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader_module,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        ComputePipeline {
            compute_pipeline,
            compute_bind_group_layout,
        }
    }

    pub fn compute(
        &self,
        compute_pass: &mut wgpu::ComputePass,
        compute_bind_group: &wgpu::BindGroup,
    ) {
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, compute_bind_group, &[]);
        compute_pass.dispatch_workgroups(16, 1, 1);
    }
}
