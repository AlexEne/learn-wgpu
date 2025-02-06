use glam::Vec3;

use crate::{
    model::Vertex, pipelines::DebugModel, renderer::Renderer, LightUniform
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightVertex {
    pub position: [f32; 3],
}

pub struct LightModel {
    pub position: Vec3,

    pub light_uniform: LightUniform,
    pub vertices: Vec<LightVertex>,
    pub indices: Vec<u16>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub light_object_transform_buffer: wgpu::Buffer,
    pub light_obj_transform_bind_group: wgpu::BindGroup,

    pub light_bind_group: wgpu::BindGroup,
}

impl LightModel {
    pub fn new(renderer: &Renderer) -> LightModel {
        let position = Vec3::new(0.0, 0.3, 0.3);
        let data = glam::Mat4::from_translation(position).to_cols_array();

        let light_object_transform_buffer = renderer.create_buffer_init(
            "Light object transform buffer",
            bytemuck::cast_slice(data.as_ref()),
            wgpu::BufferUsages::UNIFORM,
        );

        // I copied this from the DebugPipeline code.
        // TODO don't copy this :D
        let object_transform_bind_group_layout =
            renderer
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let light_obj_transform_bind_group = renderer.create_bind_group_for_buffers(
            "Light object transform bind group",
            &object_transform_bind_group_layout,
            &[&light_object_transform_buffer],
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

        let vertex_buffer = renderer.create_buffer_init(
            "Light Vertex Buffer",
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = renderer.create_buffer_init(
            "Light Index Buffer",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );

        let light_uniform = LightUniform {
            position: position.into(),
            _padding: 0.0,
            color: [1.0, 0.7, 0.7],
            _padding2: 0.0,
        };

        let light_buffer = renderer.create_buffer_init(
            "Light Buffer",
            bytemuck::cast_slice(&[light_uniform]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let light_bind_group = renderer.create_bind_group_for_buffers(
            "Light bind group",
            &renderer.light_bind_group_layout,
            &[&light_buffer],
        );

        LightModel {
            vertices,
            indices,
            light_uniform,
            vertex_buffer,
            index_buffer,
            position,
            light_object_transform_buffer,
            light_obj_transform_bind_group,
            light_bind_group,
        }
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

impl DebugModel for LightModel {
    fn object_transform_bind_group(&self) -> &wgpu::BindGroup {
        &self.light_obj_transform_bind_group
    }

    fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    fn index_format(&self) -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }

    fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }
}
