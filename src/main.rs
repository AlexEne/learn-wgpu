mod camera;
mod model;
mod shader_compiler;
mod texture;
use std::borrow::Cow;

use camera::{Camera, CameraGraphicsObject, CameraUniform};
use glam::{Quat, Vec3};
mod light;
use light::{LightModel, LightVertex};
use model::{Model, ModelVertex, Vertex};
use wgpu::{
    include_spirv_raw,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, DepthStencilState, Features,
    InstanceDescriptor, Limits, MemoryHints, MultisampleState, PipelineCompilationOptions,
    PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, ShaderSource,
    ShaderStages, SurfaceConfiguration, SurfaceError, TextureViewDescriptor, VertexAttribute,
    VertexBufferLayout,
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

    pipeline: RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    diffuse_binding_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,

    camera: Camera,
    camera_controller: CameraController,
    camera_graphics_object: CameraGraphicsObject,

    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,

    depth_texture: texture::Texture,

    model: Model,

    light_buffer: wgpu::Buffer,
    light_uniform: LightUniform,
    light_bind_group: wgpu::BindGroup,

    light_model: LightModel,
}

struct CameraController {
    position: glam::Vec3,
    speed: f32,
}

impl CameraController {
    fn new() -> CameraController {
        CameraController {
            position: glam::Vec3::new(0.0, 1.0, 2.0),
            speed: 0.05,
        }
    }

    fn process_event(&mut self, event: &WindowEvent) {
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

                self.position += movement * self.speed;
            }
            _ => {}
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        camera.position = self.position;
    }
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

    fn create_pipeline(
        device: &wgpu::Device,
        config: &SurfaceConfiguration,
        vertex_shader: wgpu::ShaderModule,
        fragment_shader: wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_group_layout: &wgpu::BindGroupLayout,
        light_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[
                bind_group_layout,
                camera_group_layout,
                light_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),

            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"), // None selects the only entry point for @vertex. Expects only one!!
                buffers: &[ModelVertex::desc(), Instance::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"), // None selects the only entry point for @fragment. Expects only one!!
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),

            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        pipeline
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

        let (vertex_shader, fragment_shader) = create_shaders(&device);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

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

        let pipeline = State::create_pipeline(
            &device,
            &config,
            vertex_shader,
            fragment_shader,
            &bind_group_layout,
            &camera_graphics_object.bind_group_layout,
            &light_bind_group_layout,
        );

        let (vertex_buffer, index_buffer) = model.create_buffers(&device);

        let instances = create_instances();
        let instances_data: Vec<InstanceRaw> = instances.iter().map(Instance::to_raw).collect();

        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let diffuse_binding_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &config);

        State {
            pipeline,
            surface,
            device,
            queue,
            config,
            size,
            window,
            vertex_buffer,
            index_buffer,
            diffuse_binding_group,
            diffuse_texture,
            camera,
            camera_controller: CameraController::new(),
            instances,
            instance_buffer,
            light_buffer,
            light_uniform: light,
            light_bind_group,
            depth_texture,
            model,
            light_model,
            camera_graphics_object,
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
        self.camera_controller.process_event(event);
        false
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_graphics_object
            .update(&self.queue, self.camera.build_uniform());

        for instance in self.instances.iter_mut() {
            instance.rotation *= Quat::from_rotation_y(0.02);
        }

        let instances_data: Vec<InstanceRaw> =
            self.instances.iter().map(Instance::to_raw).collect();
        self.queue.write_buffer(
            &self.instance_buffer,
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

            // computerrender_pass.set_vertex_buffer(slot, buffer_slice);
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.diffuse_binding_group, &[]);
            render_pass.set_bind_group(1, &self.camera_graphics_object.bind_group, &[]);
            render_pass.set_bind_group(2, &self.light_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.model.num_indices(), 0, 0..self.instances.len() as _);

            self.light_model
                .render(&mut render_pass, &self.camera_graphics_object.bind_group);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_instances() -> Vec<Instance> {
    const INSTANCES_PER_ROW: u32 = 10;
    const INSTANCE_DISPLACEMENT: Vec3 = Vec3::new(
        INSTANCES_PER_ROW as f32 * 0.5,
        0.0,
        INSTANCES_PER_ROW as f32 * 0.5,
    );

    (0..INSTANCES_PER_ROW)
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
        .collect()
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

    event_loop.run(move |event, control_flow| match event {
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
                state.update();

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
    })
}

fn main() -> Result<(), EventLoopError> {
    run()
}
