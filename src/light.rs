use glam::Vec3;
use gltf::{camera, Buffer};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    ShaderModuleDescriptor, SurfaceConfiguration,
};

use crate::{
    model::Vertex,
    shader_compiler::{self, compile_shaders},
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightVertex {
    pub position: [f32; 3],
}

pub struct LightModel {
    pub position: Vec3,

    pub vertices: Vec<LightVertex>,
    pub indices: Vec<u16>,
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_shader: wgpu::ShaderModule,
    pub fragment_shader: wgpu::ShaderModule,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub light_object_transform_buffer: wgpu::Buffer,
    pub light_obj_transform_bind_group: wgpu::BindGroup,
}

impl LightModel {
    pub fn new(
        device: &wgpu::Device,
        camera_group_layout: &wgpu::BindGroupLayout,
        config: &SurfaceConfiguration,
    ) -> LightModel {
        let (vertex_shader, fragment_shader) = compile_shaders(
            shader_compiler::ShaderInput {
                shader_code: include_str!("shaders/light.vert"),
                entry_point: "main",
                file_name: "light.vert",
            },
            shader_compiler::ShaderInput {
                shader_code: include_str!("shaders/light.frag"),
                entry_point: "main",
                file_name: "light.frag",
            },
        );

        let vertex_shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Light Vertex Shader"),
            source: wgpu::util::make_spirv(&vertex_shader.as_binary_u8()),
        });

        let fragment_shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Light Fragment Shader"),
            source: wgpu::util::make_spirv(&fragment_shader.as_binary_u8()),
        });

        let object_transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera object model bind group layout"),
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
 
        let position = Vec3::new(0.0, 0.3, 0.3);
        let data = glam::Mat4::from_translation(position).to_cols_array();
        println!("{:?}", data);
        let light_object_transform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Light object transform buffer"),
            contents: bytemuck::cast_slice(
                    data
                    .as_ref(),
            ),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let light_obj_transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &object_transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_object_transform_buffer.as_entire_binding(),
            }],
            label: Some("Light object transform bind group"),
        });

        let pipeline = LightModel::create_light_render_pipeline(
            &device,
            &camera_group_layout,
            &object_transform_bind_group_layout,
            &vertex_shader_module,
            &fragment_shader_module,
            &config,
        );

        let vertices = vec![
            LightVertex {
                position: [-0.05, -0.05, -0.05],
            },
            LightVertex {
                position: [0.05, -0.05, -0.05],
            },
            LightVertex {
                position: [0.05, 0.05, -0.05],
            },
            LightVertex {
                position: [-0.05, 0.05, -0.05],
            },
            LightVertex {
                position: [-0.05, -0.05, 0.05],
            },
            LightVertex {
                position: [0.05, -0.05, 0.05],
            },
            LightVertex {
                position: [0.05, 0.05, 0.05],
            },
            LightVertex {
                position: [-0.05, 0.05, 0.05],
            },
        ];
        let indices = vec![
            0, 1, 2, 2, 3, 0, // front face
            4, 5, 6, 6, 7, 4, // back face
            0, 1, 5, 5, 4, 0, // bottom face
            2, 3, 7, 7, 6, 2, // top face
            0, 3, 7, 7, 4, 0, // left face
            1, 2, 6, 6, 5, 1, // right face
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        LightModel {
            vertices,
            indices,
            pipeline,
            vertex_shader: vertex_shader_module,
            fragment_shader: fragment_shader_module,
            vertex_buffer,
            index_buffer,
            position,
            light_object_transform_buffer,
            light_obj_transform_bind_group,
        }
    }

    fn create_light_render_pipeline(
        device: &wgpu::Device,
        camera_group_layout: &wgpu::BindGroupLayout,
        object_transform_bind_group_layout: &wgpu::BindGroupLayout,
        vertex_shader_module: &wgpu::ShaderModule,
        fragment_shader_module: &wgpu::ShaderModule,
        config: &SurfaceConfiguration,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[camera_group_layout, object_transform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Light Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader_module,
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
                module: &fragment_shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        pipeline
    }

    pub fn render(&self, render_pass: &mut wgpu::RenderPass, camera_bind_group: &wgpu::BindGroup) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_obj_transform_bind_group, &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
    }
}

impl Vertex for LightVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![
            0 => Float32x3,
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LightVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}
