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
    pub far: f32,
}

impl Camera {
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
