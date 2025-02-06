use crate::{camera::CameraGraphicsObject, light::LightVertex, model::Vertex, shader_compiler};

/// Not PBR, just uses a single color
pub struct DebugPipeline {
    pipeline: wgpu::RenderPipeline,
    object_transform_bind_group_layout: wgpu::BindGroupLayout,
}

pub trait DebugModel {
    fn object_transform_bind_group(&self) -> &wgpu::BindGroup;
    fn vertex_buffer(&self) -> &wgpu::Buffer;
    fn index_buffer(&self) -> &wgpu::Buffer;
    fn index_format(&self) -> wgpu::IndexFormat;
    fn num_indices(&self) -> u32;
}

impl DebugPipeline {
    pub fn new(
        device: &wgpu::Device,
        fragment_output_texture_format: wgpu::TextureFormat,
        camera_graphics_object: &CameraGraphicsObject,
    ) -> DebugPipeline {
        let (vertex_shader, fragment_shader) = shader_compiler::compile_shaders(
            device,
            shader_compiler::ShaderInput {
                shader_code: include_str!("../shaders/simple_transform.vert"),
                entry_point: "main",
                file_name: "light.vert",
            },
            shader_compiler::ShaderInput {
                shader_code: include_str!("../shaders/simple_frag.frag"),
                entry_point: "main",
                file_name: "light.frag",
            },
        );

        let object_transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Debug pipeline transform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Debug pipeline layout"),
            bind_group_layouts: &[
                &camera_graphics_object.bind_group_layout,
                &object_transform_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[LightVertex::desc()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fragment_output_texture_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        DebugPipeline {
            pipeline,
            object_transform_bind_group_layout,
        }
    }

    pub fn draw_indexed(
        &self,
        render_pass: &mut wgpu::RenderPass,
        debug_material_instance: DebugMaterialInstance,
        debug_model: &dyn DebugModel,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, debug_material_instance.camera_bind_group, &[]);
        render_pass.set_bind_group(1, debug_model.object_transform_bind_group(), &[]);
        render_pass.set_vertex_buffer(0, debug_model.vertex_buffer().slice(..));
        render_pass.set_index_buffer(
            debug_model.index_buffer().slice(..),
            debug_model.index_format(),
        );
        render_pass.draw_indexed(0..debug_model.num_indices(), 0, 0..1);
    }
}

pub struct DebugMaterialInstance<'a> {
    pub camera_bind_group: &'a wgpu::BindGroup,
}
