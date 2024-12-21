use bytemuck;
use glam::Vec3;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, Device,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub cam: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> CameraUniform {
        CameraUniform {
            cam: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

pub struct Camera {
    pub position: Vec3,
    pub center: Vec3,
    pub up: Vec3,

    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
}

pub struct CameraGraphicsObject {
    pub uniform_buffer: Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CameraGraphicsObject {
    pub fn new(device: &wgpu::Device) -> CameraGraphicsObject {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        CameraGraphicsObject {
            uniform_buffer,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, camera_uniform: CameraUniform) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }
}

impl Camera {
    pub fn new(
        position: Vec3,
        center: Vec3,
        up: Vec3,
        fov: f32,
        aspect_ratio: f32,
        near: f32,
    ) -> Camera {
        Camera {
            position,
            center,
            up,
            fov,
            aspect_ratio,
            near,
        }
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.position, self.center, self.up);
        let projection =
            glam::Mat4::perspective_infinite_reverse_rh(self.fov, self.aspect_ratio, self.near);
        projection * view
    }

    pub fn build_uniform(&self) -> CameraUniform {
        let view_projection = self.build_view_projection_matrix();
        CameraUniform {
            cam: view_projection.to_cols_array_2d(),
        }
    }
}
