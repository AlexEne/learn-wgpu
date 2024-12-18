mod texture;

use wgpu::{
    include_spirv_raw,
    util::{BufferInitDescriptor, DeviceExt},
    BlendState, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Features,
    InstanceDescriptor, Limits, MemoryHints, MultisampleState, PipelineCompilationOptions,
    PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SurfaceError, TextureViewDescriptor, VertexAttribute,
    VertexBufferLayout,
};
use winit::{
    error::EventLoopError,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

const VERTICES: &[Vertex] = &[
    // Changed
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        tex_coords: [0.4131759, 0.00759614],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        tex_coords: [0.0048659444, 0.43041354],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        tex_coords: [0.28081453, 0.949397],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        tex_coords: [0.85967, 0.84732914],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        tex_coords: [0.9414737, 0.2652641],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

impl Vertex {
    const ATTRIBUTES: [VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

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
    num_vertices: u32,

    index_buffer: wgpu::Buffer,
    num_indices: u32,

    diffuse_binding_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        // let compiler = shaderc::Compiler::new().unwrap();
        // let vertex_shader = compiler
        //     .compile_into_spirv(
        //         include_str!("shaders-glsl/vertex_shader.vert"),
        //         shaderc::ShaderKind::Vertex,
        //         "shader.vert",
        //         "main",
        //         None,
        //     )
        //     .unwrap();

        // let fragment_shader = compiler
        //     .compile_into_spirv(
        //         include_str!("shaders-glsl/fragment.frag"),
        //         shaderc::ShaderKind::Fragment,
        //         "fragment.frag",
        //         "main",
        //         None,
        //     )
        //     .unwrap();

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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: Features::SPIRV_SHADER_PASSTHROUGH,
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
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

        // let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let vertex_shader = unsafe {
            device.create_shader_module_spirv(&include_spirv_raw!("../data/spirv/glsl/vertex.spv"))
        };
        let fragment_shader = unsafe {
            device
                .create_shader_module_spirv(&include_spirv_raw!("../data/spirv/glsl/fragment.spv"))
        };

        // let vertex_shader = device.create_shader_module(ShaderModuleDescriptor {
        //     label: Some("Vtx Shader"),
        //     source: ShaderSource::Glsl {
        //         shader: Cow::from(include_str!("shaders-glsl/vertex_shader.vert")),
        //         stage: wgpu::naga::ShaderStage::Vertex,
        //         defines: Default::default(),
        //     },
        // });

        // let fragment_shader = device.create_shader_module(ShaderModuleDescriptor {
        //     label: Some("Frag Shader"),
        //     source: ShaderSource::Glsl {
        //         shader: Cow::from(include_str!("shaders-glsl/fragment.frag")),
        //         stage: wgpu::naga::ShaderStage::Fragment,
        //         defines: Default::default(),
        //     },
        // });

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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        println!("{:?}", config.format);
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),

            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"), // None selects the only entry point for @vertex. Expects only one!!
                buffers: &[Vertex::desc()],
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
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
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

        State {
            pipeline,
            surface,
            device,
            queue,
            config,
            size,
            window,
            vertex_buffer,
            num_vertices: VERTICES.len() as u32,
            index_buffer,
            num_indices: INDICES.len() as u32,
            diffuse_binding_group,
            diffuse_texture,
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
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

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
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // computerrender_pass.set_vertex_buffer(slot, buffer_slice);
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.diffuse_binding_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
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
