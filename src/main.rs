mod camera;
mod material;
mod model;
mod shader_compiler;
mod texture;
use std::{borrow::Cow, time};

use camera::{Camera, CameraGraphicsObject};
use glam::{Quat, Vec3};
mod light;
use light::LightModel;
use material::{PBRMaterial, PBRMaterialInstance};
use model::{Model, ModelGPUData, ModelGPUDataInstanced};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Color,
    CommandEncoderDescriptor, Features, InstanceDescriptor, Limits, MemoryHints,
    RenderPassColorAttachment, RenderPassDescriptor, ShaderStages, SurfaceError,
    TextureViewDescriptor, VertexAttribute, VertexBufferLayout,
};
use winit::{
    error::EventLoopError,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,

    pbr_material: PBRMaterial,

    textures_binding_group: wgpu::BindGroup,
    base_color: texture::Texture,
    metalic_roughness: texture::Texture,

    camera: Camera,
    camera_graphics_object: CameraGraphicsObject,

    depth_texture: texture::Texture,

    model: Model,

    light_buffer: wgpu::Buffer,
    light_uniform: LightUniform,
    light_bind_group: wgpu::BindGroup,

    pbr_factors_bind_group: wgpu::BindGroup,
    pbr_factors_buffer: wgpu::Buffer,

    light_model: LightModel,

    instances: Vec<Instance>,
    model_gpu_instanced: ModelGPUDataInstanced,
}

struct Instance {
    position: Vec3,
    rotation: Quat,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model_mtx: (glam::Mat4::from_translation(self.position)
                * glam::Mat4::from_quat(self.rotation)
                * glam::Mat4::from_scale(Vec3::splat(10.0)))
            .to_cols_array_2d(),
        }
    }

    fn desc() -> VertexBufferLayout<'static> {
        const ATTRIBUTES: [VertexAttribute; 4] = wgpu::vertex_attr_array![5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4];
        VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model_mtx: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    _padding: f32,
    color: [f32; 3],
    _padding2: f32,
}

impl<'a> State<'a> {
    fn configure_surface(
        surface: &wgpu::Surface,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> wgpu::SurfaceConfiguration {
        let surface_caps = surface.get_capabilities(adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(device, &config);
        config
    }

    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        let model = Model::from_gltf("data/Avocado.glb");

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        #[cfg(windows)]
        let required_features = Features::SPIRV_SHADER_PASSTHROUGH;
        #[cfg(not(windows))]
        let required_features = Features::empty();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features,
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();

        let config = State::configure_surface(&surface, &adapter, &device, size);

        let base_color = texture::Texture::from_bytes(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &model.material.base_color_texture,
            "Base Color",
        )
        .unwrap();

        let metalic_roughness = texture::Texture::from_bytes(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &model.material.metalic_roughness_texture,
            "Metalic Roughness",
        )
        .unwrap();

        let camera = Camera::new(
            glam::Vec3::new(0.0, 1.0, 2.0),
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::Y,
            45.0_f32.to_radians(),
            config.width as f32 / config.height as f32,
            0.1,
        );
        let camera_graphics_object = CameraGraphicsObject::new(&device);

        let light_model =
            LightModel::new(&device, &camera_graphics_object.bind_group_layout, &config);
        let light = LightUniform {
            position: light_model.position.into(),
            _padding: 0.0,
            color: [1.0, 0.7, 0.7],
            _padding2: 0.0,
        };

        let light_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Light buffer bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        let pbr_factors_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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

        let pbr_factors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBR Factors Buffer"),
            contents: bytemuck::cast_slice(&[model.material.pbr_factors]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pbr_factors_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("PBR Factors Bind Group"),
            layout: &pbr_factors_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pbr_factors_buffer.as_entire_binding(),
            }],
        });

        let pbr_material = material::PBRMaterial::new(
            &device,
            config.format,
            &camera_graphics_object.bind_group_layout,
            &light_bind_group_layout,
            &pbr_factors_bind_group_layout,
        );

        let textures_binding_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse Bind Group"),
            layout: &pbr_material.textures_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_color.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&base_color.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&metalic_roughness.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&metalic_roughness.sampler),
                },
            ],
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &config);

        let (model_gpu_instanced, instances) =
            create_instances(&device, model.create_gpu_data(&device));

        State {
            pbr_material,
            surface,
            device,
            queue,
            config,
            size,
            window,

            textures_binding_group,
            base_color,
            metalic_roughness,

            pbr_factors_bind_group,
            pbr_factors_buffer,

            camera,
            light_buffer,
            light_uniform: light,
            light_bind_group,
            depth_texture,
            model,
            light_model,
            camera_graphics_object,

            instances,
            model_gpu_instanced,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_event(event);
        false
    }

    fn update(&mut self, dt: time::Duration) {
        self.camera.update(dt);
        self.camera_graphics_object
            .update(&self.queue, self.camera.build_uniform());

        for instance in self.instances.iter_mut() {
            instance.rotation *= Quat::from_rotation_y(0.02);
        }

        let instances_data: Vec<InstanceRaw> =
            self.instances.iter().map(Instance::to_raw).collect();
        self.queue.write_buffer(
            &self.model_gpu_instanced.instance_buffer,
            0,
            bytemuck::cast_slice(&instances_data),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Main encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass 1"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.pbr_material.draw_instanced(
                &mut render_pass,
                PBRMaterialInstance {
                    textures_bind_group: &self.textures_binding_group,
                    camera_bind_group: &self.camera_graphics_object.bind_group,
                    light_bind_group: &self.light_bind_group,
                    pbr_factors_bind_group: &self.pbr_factors_bind_group,
                },
                &self.model_gpu_instanced,
            );

            self.light_model
                .render(&mut render_pass, &self.camera_graphics_object.bind_group);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_instances(
    device: &wgpu::Device,
    model_data: ModelGPUData,
) -> (ModelGPUDataInstanced, Vec<Instance>) {
    const INSTANCES_PER_ROW: u32 = 10;
    const INSTANCE_DISPLACEMENT: Vec3 = Vec3::new(
        INSTANCES_PER_ROW as f32 * 0.5,
        0.0,
        INSTANCES_PER_ROW as f32 * 0.5,
    );

    let instances: Vec<Instance> = (0..INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..INSTANCES_PER_ROW).map(move |x| {
                let position = glam::Vec3::new(x as f32, 0.0, z as f32);

                let rotation;
                if position == glam::Vec3::ZERO {
                    rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, 0.0);
                } else {
                    rotation = glam::Quat::from_axis_angle(
                        position.normalize(),
                        std::f32::consts::PI / 4.0,
                    );
                }

                Instance { position, rotation }
            })
        })
        .collect();

    let instances_data: Vec<InstanceRaw> = instances.iter().map(Instance::to_raw).collect();

    let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instances_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    (
        ModelGPUDataInstanced {
            model_gpu_data: model_data,
            instance_buffer,
            num_instances: instances.len() as u32,
        },
        instances,
    )
}

fn create_shaders(device: &wgpu::Device) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
    let (vertex_shader, fragment_shader) = shader_compiler::compile_shaders(
        shader_compiler::ShaderInput {
            shader_code: include_str!("shaders/vertex_shader.vert"),
            file_name: "vertex_shader.vert",
            entry_point: "main",
        },
        shader_compiler::ShaderInput {
            shader_code: include_str!("shaders/fragment.frag"),
            file_name: "fragment.frag",
            entry_point: "main",
        },
    );

    #[cfg(not(target_os = "macos"))]
    let (vertex_shader, fragment_shader) = {
        let vertex_shader = unsafe {
            device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                label: Some("Vertex Shader"),
                source: Cow::from(vertex_shader.as_binary()),
            })
        };
        let fragment_shader = unsafe {
            device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                label: Some("Fragment Shader"),
                source: Cow::from(fragment_shader.as_binary()),
            })
        };

        (vertex_shader, fragment_shader)
    };

    #[cfg(target_os = "macos")]
    let (vertex_shader, fragment_shader) = {
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(vertex_shader.as_binary())),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(fragment_shader.as_binary())),
        });

        (vertex_shader, fragment_shader)
    };
    (vertex_shader, fragment_shader)
}

pub fn run() -> Result<(), EventLoopError> {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = pollster::block_on(State::new(&window));
    let mut last_update = time::Instant::now();

    event_loop.run(move |event, control_flow| {

        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() && !state.input(event) => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => control_flow.exit(),
                WindowEvent::Resized(new_size) => {
                    state.resize(*new_size);
                }
                WindowEvent::RedrawRequested => {
                    state.window.request_redraw();
                    
                    let dt = last_update.elapsed();
                    last_update = time::Instant::now();
                    state.update(dt);

                    match state.render() {
                        Ok(_) => (),
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            state.resize(state.size);
                        }
                        Err(SurfaceError::OutOfMemory) => control_flow.exit(),
                        Err(SurfaceError::Timeout) => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })
}

fn main() -> Result<(), EventLoopError> {
    run()
}
