mod camera;
mod model;
mod pipelines;
mod renderer;
mod shader_compiler;
mod texture;
use std::time;

use camera::Camera;
use glam::{Quat, Vec3};
mod light;
use light::LightModel;
use model::{Model, ModelGPUData, ModelGPUDataInstanced};
use renderer::Renderer;
use wgpu::{
    util::{DeviceExt, DrawIndexedIndirectArgs}, InstanceDescriptor, SurfaceError, VertexAttribute, VertexBufferLayout,
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
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,

    camera: Camera,

    models: Vec<Model>,

    light_buffer: wgpu::Buffer,
    light_uniform: LightUniform,
    light_bind_group: wgpu::BindGroup,

    light_model: LightModel,
    models_instanced: Vec<InstancedModel>,

    renderer: Renderer,
}

struct InstancedModel {
    instances: Vec<Instance>,
    model_gpu_instanced: ModelGPUDataInstanced,

    // Buffer for indirect draw calls that gets filled by the compute pass
    indirect_draw_buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,

    // Bind groups for the textures and pbr factors used during render pass
    geometry_and_textures_bind_group: wgpu::BindGroup,
    pbr_factors_bind_group: wgpu::BindGroup,
    pbr_factors_buffer: wgpu::Buffer,
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
        const ATTRIBUTES: [VertexAttribute; 4] = wgpu::vertex_attr_array![
            5 => Float32x4,
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4
        ];
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
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        let mut textures = Vec::new();

        let instance = wgpu::Instance::new(&InstanceDescriptor {
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

        let renderer = Renderer::new(&adapter, &surface, (size.width, size.height));

        let mut models = Model::from_gltf(&renderer, "data/Corset.glb", &mut textures);
        let avocado = Model::from_gltf(&renderer, "data/Avocado.glb", &mut textures);
        models.extend(avocado.into_iter());

        let camera = Camera::new(
            glam::Vec3::new(0.0, 0.5, 2.0),
            glam::Vec3::new(0.0, 0.0, 0.0),
            45.0_f32.to_radians(),
            renderer.viewport_width() / renderer.viewport_height(),
            0.1,
        );

        // TODO make device private
        let light_model = LightModel::new(
            &renderer.device,
            &renderer.camera_graphics_object.bind_group_layout,
            &renderer.config,
        );
        let light = LightUniform {
            position: light_model.position.into(),
            _padding: 0.0,
            color: [1.0, 0.7, 0.7],
            _padding2: 0.0,
        };

        let light_buffer = renderer.create_buffer_init(
            "Light Buffer",
            bytemuck::cast_slice(&[light]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let light_bind_group = renderer.create_bind_group_for_buffers(
            "Light bind group",
            &renderer.light_bind_group_layout,
            &[&light_buffer],
        );

        let mut models_instanced = Vec::new();
        let mut indirect_args = Vec::new();
        let instance_output_buffer = renderer.create_buffer(
            "Instance Output Buffer",
            1024 * 1024 * 16,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        );

        for (idx, model) in models.iter().enumerate() {
            let pbr_factors_buffer = renderer.create_buffer_init(
                "PBR Factors Buffer",
                bytemuck::cast_slice(&[model.material.pbr_factors]),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            let pbr_factors_bind_group = renderer.create_bind_group_for_buffers(
                "PBR Factors Bind Group",
                &renderer.pbr_material_pipeline.pbr_factors_bind_group_layout,
                &[&pbr_factors_buffer],
            );

            let base_color = &textures[model.material.base_color_texture.0];
            let metalic_roughness = &textures[model.material.metalic_roughness_texture.0];

            let (model_gpu_instanced, instances) = create_instances(
                &renderer,
                model.create_gpu_data(&renderer),
                0.7 * idx as f32,
            );

            let geometry_and_textures_bind_group =
                renderer
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Diffuse Bind Group"),
                        layout: &renderer
                            .pbr_material_pipeline
                            .geometry_and_textures_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &model_gpu_instanced.model_gpu_data.vertex_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&base_color.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&base_color.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &metalic_roughness.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(
                                    &metalic_roughness.sampler,
                                ),
                            },
                        ],
                    });

            indirect_args.push(DrawIndexedIndirectArgs {
                index_count: model_gpu_instanced.model_gpu_data.index_buffer.num_indices,
                instance_count: model_gpu_instanced.num_instances,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            });

            let indirect_draw_buffer = renderer.create_buffer_init(
                "Indirect Buffer",
                unsafe {
                    std::slice::from_raw_parts(
                        indirect_args.as_ptr() as *const u8,
                        std::mem::size_of_val(&indirect_args),
                    )
                },
                wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
            );

            let compute_bind_group = renderer.create_bind_group_for_buffers(
                "Compute Bind Group",
                &renderer.compute_pipeline.compute_bind_group_layout,
                &[
                    &indirect_draw_buffer,
                    &model_gpu_instanced.instance_buffer,
                    &model_gpu_instanced.instance_output_buffer,
                    &model_gpu_instanced.model_gpu_data.bounding_sphere,
                ],
            );

            let instanced_model = InstancedModel {
                instances,
                model_gpu_instanced,
                geometry_and_textures_bind_group,
                pbr_factors_bind_group,
                pbr_factors_buffer,
                indirect_draw_buffer,
                compute_bind_group,
            };

            models_instanced.push(instanced_model);
        }

        State {
            surface,
            size,
            window,

            camera,
            light_buffer,
            light_uniform: light,
            light_bind_group,

            models,
            light_model,

            renderer,
            models_instanced,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.renderer.resize(new_size.width, new_size.height);
            self.surface
                .configure(&self.renderer.device, &self.renderer.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_event(event);
        false
    }

    fn update(&mut self, dt: time::Duration) {
        self.camera.update(dt);
        self.renderer
            .camera_graphics_object
            .update(&self.renderer.queue, self.camera.build_uniform());

        for instanced_model in self.models_instanced.iter_mut() {
            for instance in instanced_model.instances.iter_mut() {
                instance.rotation *= Quat::from_rotation_y(0.02);
            }

            let instances_data: Vec<InstanceRaw> = instanced_model
                .instances
                .iter()
                .map(Instance::to_raw)
                .collect();

            self.renderer.write_buffer(
                &instanced_model.model_gpu_instanced.instance_buffer,
                0,
                bytemuck::cast_slice(&instances_data),
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.renderer.render(
            &self.surface,
            &self.models_instanced,
            &self.light_bind_group,
            &self.light_model,
        )
    }
}

fn create_instances(
    renderer: &Renderer,
    model_data: ModelGPUData,
    y_offset: f32,
) -> (ModelGPUDataInstanced, Vec<Instance>) {
    const INSTANCES_PER_ROW: u32 = 10;

    let instances: Vec<Instance> = (0..INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..INSTANCES_PER_ROW).map(move |x| {
                let position = glam::Vec3::new(x as f32, y_offset, z as f32);

                let rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, 0.0);

                Instance { position, rotation }
            })
        })
        .collect();

    let instances_data: Vec<InstanceRaw> = instances.iter().map(Instance::to_raw).collect();

    let instance_buffer = renderer.create_buffer_init(
        "Instance Buffer",
        bytemuck::cast_slice(&instances_data),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    );

    let instance_output_buffer = renderer.create_buffer(
        "Instance Output Buffer",
        1024 * 1024 * 16,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    );

    (
        ModelGPUDataInstanced {
            model_gpu_data: model_data,
            instance_buffer,
            instance_output_buffer,
            num_instances: instances.len() as u32,
        },
        instances,
    )
}

pub fn run() -> Result<(), EventLoopError> {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = pollster::block_on(State::new(&window));
    let mut last_update = time::Instant::now();

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

                let dt = last_update.elapsed();
                last_update = time::Instant::now();
                state.update(dt);

                match state.render() {
                    Ok(_) => (),
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size);
                    }
                    Err(SurfaceError::OutOfMemory) => control_flow.exit(),
                    Err(SurfaceError::Timeout | SurfaceError::Other) => {}
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
