use glam::Vec4;
use wgpu::{MultisampleState, PipelineLayoutDescriptor, TextureFormat};

use crate::{
    model::{ModelGPUDataInstanced, Vertex},
    shader_compiler, Instance,
};

#[derive(Clone, Copy, Debug)]
pub struct TextureID(pub usize);

pub struct MaterialData {
    pub base_color_texture: TextureID,
    pub metalic_roughness_texture: TextureID,
    pub pbr_factors: PBRFactors,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PBRFactors {
    pub base_color_factor: [f32; 4],
    pub metalic_factor: f32,
    pub roughness_factor: f32,
    pub _padding: [f32; 2],
}

impl MaterialData {
    pub fn new(
        base_color_texture: TextureID,
        metalic_roughness_texture: TextureID,
        metalic_factor: f32,
        roughness_factor: f32,
        base_color_factor: Vec4,
    ) -> MaterialData {
        MaterialData {
            base_color_texture,
            metalic_roughness_texture,
            pbr_factors: PBRFactors {
                metalic_factor,
                roughness_factor,
                base_color_factor: base_color_factor.into(),
                _padding: [0.0; 2],
            },
        }
    }
}

pub struct PBRMaterialPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub geometry_and_textures_bind_group_layout: wgpu::BindGroupLayout,
    pub pbr_factors_bind_group_layout: wgpu::BindGroupLayout,
}

impl PBRMaterialPipeline {
    pub fn new(
        device: &wgpu::Device,
        output_format: TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        light_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> PBRMaterialPipeline {
        let pbr_factors_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR Factors bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let geometry_and_textures_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR Textures bind group layout"),
                entries: &[
                    // Geometry
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Base Color Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Metalic Roughness Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let (vertex_shader, fragment_shader) = shader_compiler::compile_shaders(
            device,
            shader_compiler::ShaderInput {
                shader_code: include_str!("../shaders/vertex_shader.vert"),
                file_name: "vertex_shader.vert",
                entry_point: "main",
            },
            shader_compiler::ShaderInput {
                shader_code: include_str!("../shaders/fragment.frag"),
                file_name: "fragment.frag",
                entry_point: "main",
            },
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("PBR Material Pipeline Layout"),
            bind_group_layouts: &[
                &geometry_and_textures_bind_group_layout,
                camera_bind_group_layout,
                light_bind_group_layout,
                &pbr_factors_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR Material Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Instance::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater, //Reverse depth test
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview: None,
        });

        PBRMaterialPipeline {
            pipeline,
            geometry_and_textures_bind_group_layout,
            pbr_factors_bind_group_layout,
        }
    }

    pub fn draw_instanced(
        &self,
        render_pass: &mut wgpu::RenderPass,
        material_instance: PBRMaterialInstance,
        model_gpu_instanced: &ModelGPUDataInstanced,
        indirect_draw_buffer: &wgpu::Buffer,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        // render_pass.set_bind_group(0, material_instance.geometry_bind_group, &[]);
        render_pass.set_bind_group(0, material_instance.geometry_and_textures_bind_group, &[]);
        render_pass.set_bind_group(1, material_instance.camera_bind_group, &[]);
        render_pass.set_bind_group(2, material_instance.light_bind_group, &[]);
        render_pass.set_bind_group(3, material_instance.pbr_factors_bind_group, &[]);

        let model_gpu_data = &model_gpu_instanced.model_gpu_data;
        let index_buffer = &model_gpu_data.index_buffer;
        render_pass.set_vertex_buffer(0, model_gpu_instanced.instance_output_buffer.slice(..));
        render_pass.set_index_buffer(
            index_buffer.index_buffer.slice(..),
            index_buffer.index_buffer_format,
        );

        render_pass.multi_draw_indexed_indirect(indirect_draw_buffer, 0, 1);
    }
}

// This is constructed in place when calling draw_instanced
// This is because I might want to reuse bind groups across pipelines.
// So as long as they respect the bind group layout, this works fine for now.
pub struct PBRMaterialInstance<'a> {
    pub geometry_and_textures_bind_group: &'a wgpu::BindGroup,
    pub camera_bind_group: &'a wgpu::BindGroup,
    pub light_bind_group: &'a wgpu::BindGroup,
    pub pbr_factors_bind_group: &'a wgpu::BindGroup,
}
