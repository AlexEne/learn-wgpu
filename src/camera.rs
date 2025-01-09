use std::time;

use bytemuck;
use dolly::prelude::*;
use glam::{Quat, Vec3, Vec4};
use gltf::{
    animation::Target,
    camera::{self, Projection},
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, Device,
};
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub cam: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> CameraUniform {
        CameraUniform {
            cam: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: Vec4::ZERO.into(),
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
    camera_rig: CameraRig,
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
        let camera_rig = CameraRig::builder()
            .with(Position::new([position.x, position.y, position.z]))
            .with(LookAt::new([center.x, center.y, center.z]))
            .with(YawPitch::new())
            .with(Smooth::new_position_rotation(1.25, 1.25))
            .build();

        Camera {
            position,
            center,
            up,
            fov,
            aspect_ratio,
            near,
            camera_rig,
        }
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        // let view = glam::Mat4::look_at_rh(self.position, self.center, self.up);
        // let projection =
        //     glam::Mat4::perspective_infinite_reverse_rh(self.fov, self.aspect_ratio, self.near);
        // projection * view

        // let view = self.camera_rig.final_transform;

        let transform = self.camera_rig.final_transform;
        let view = glam::Mat4::look_at_rh(
            transform.position.into(),
            transform.forward::<Vec3>(),
            transform.up::<Vec3>(),
        );

        let projection =
            glam::Mat4::perspective_infinite_reverse_rh(self.fov, self.aspect_ratio, self.near);

        projection * view
    }

    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state,
                        ..
                    },
                ..
            } => {
                if state == &ElementState::Released {
                    return;
                }
                let movement = match keycode {
                    KeyCode::KeyW => glam::Vec3::new(0.0, 0.0, -1.0),
                    KeyCode::KeyS => glam::Vec3::new(0.0, 0.0, 1.0),
                    KeyCode::KeyA => glam::Vec3::new(-1.0, 0.0, 0.0),
                    KeyCode::KeyD => glam::Vec3::new(1.0, 0.0, 0.0),
                    KeyCode::Space => glam::Vec3::new(0.0, 1.0, 0.0),
                    KeyCode::ControlLeft => glam::Vec3::new(0.0, -1.0, 0.0),
                    _ => glam::Vec3::ZERO,
                };

                let driver = self.camera_rig.driver_mut::<YawPitch>();
                driver.rotate_yaw_pitch(10.0, 0.0);
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: time::Duration) {
        self.camera_rig.update(dt.as_secs_f32());
    }

    pub fn build_uniform(&self) -> CameraUniform {
        let view_projection = self.build_view_projection_matrix();
        CameraUniform {
            cam: view_projection.to_cols_array_2d(),
            camera_pos: self.center.extend(0.0).into(),
        }
    }
}
