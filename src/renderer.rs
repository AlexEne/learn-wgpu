use pollster::block_on;
use wgpu::{util::DeviceExt, Color};

use crate::{
    camera::CameraGraphicsObject,
    light::LightModel,
    pipelines::{
        self, ComputePipeline, DebugMaterialInstance, DebugPipeline, PBRMaterialInstance,
        PBRMaterialPipeline,
    },
    texture::{self, Texture},
    InstancedModel,
};

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub pbr_material_pipeline: PBRMaterialPipeline,
    pub compute_pipeline: ComputePipeline,
    pub debug_pipeline: DebugPipeline,

    pub textures: Vec<texture::Texture>,
    depth_texture: texture::Texture,

    pub config: wgpu::SurfaceConfiguration,

    pub light_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_graphics_object: CameraGraphicsObject,
}

impl Renderer {
    pub fn new(adapter: &wgpu::Adapter, surface: &wgpu::Surface, size: (u32, u32)) -> Renderer {
        #[cfg(windows)]
        let required_features =
            wgpu::Features::SPIRV_SHADER_PASSTHROUGH | Features::MULTI_DRAW_INDIRECT;
        #[cfg(not(windows))]
        let required_features = wgpu::Features::MULTI_DRAW_INDIRECT;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .unwrap();

        let camera_graphics_object = CameraGraphicsObject::new(&device);

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light buffer bind group layout"),
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

        let config = configure_surface(&surface, &adapter, &device, size);

        let pbr_material_pipeline = pipelines::PBRMaterialPipeline::new(
            &device,
            config.format,
            &camera_graphics_object.bind_group_layout,
            &light_bind_group_layout,
        );

        let compute_pipeline =
            pipelines::ComputePipeline::new(&device, &camera_graphics_object.bind_group_layout);

        let debug_pipeline = DebugPipeline::new(&device, config.format, &camera_graphics_object);

        let depth_texture =
            Renderer::create_depth_texture(&device, config.width.max(1), config.height.max(1));

        Renderer {
            device,
            queue,

            pbr_material_pipeline,
            compute_pipeline,
            debug_pipeline,

            light_bind_group_layout,

            config,
            camera_graphics_object,

            textures: Vec::new(),
            depth_texture,
        }
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> Texture {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Texture {
            // texture,
            view,
            sampler,
        }
    }

    pub fn viewport_width(&self) -> f32 {
        self.config.width as _
    }

    pub fn viewport_height(&self) -> f32 {
        self.config.height as _
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.config.width = width;
        self.config.height = height;
        // self.surface.configure(&self.device, &self.config);
        self.depth_texture = Renderer::create_depth_texture(&self.device, width, height);
    }

    pub fn create_buffer_init(
        &self,
        name: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents,
                usage,
            })
    }

    pub fn create_buffer(
        &self,
        label: &str,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: wgpu::BufferAddress, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    pub fn create_bind_group_for_buffers(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    pub fn render(
        &mut self,
        surface: &wgpu::Surface,
        models_instanced: &[InstancedModel],
        light_model: &LightModel,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut compute_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute encoder"),
                });

        let mut compute_pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute pass"),
            timestamp_writes: None,
        });

        let mut render_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });
        let mut render_pass = render_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass 1"),
            color_attachments: &[
                // This is what @location(0) in the fragment shader targets
                Some(wgpu::RenderPassColorAttachment {
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

        for instanced_model in models_instanced.iter() {
            self.compute_pipeline.compute(
                &mut compute_pass,
                &instanced_model.compute_bind_group,
                &self.camera_graphics_object.bind_group,
            );

            self.pbr_material_pipeline.draw_instanced(
                &mut render_pass,
                PBRMaterialInstance {
                    geometry_and_textures_bind_group: &instanced_model
                        .geometry_and_textures_bind_group,
                    camera_bind_group: &self.camera_graphics_object.bind_group,
                    light_bind_group: &light_model.light_bind_group,
                    pbr_factors_bind_group: &instanced_model.pbr_factors_bind_group,
                },
                &instanced_model.model_gpu_instanced,
                &instanced_model.indirect_draw_buffer,
            );
        }

        self.debug_pipeline.draw_indexed(
            &mut render_pass,
            DebugMaterialInstance {
                camera_bind_group: &self.camera_graphics_object.bind_group,
            },
            light_model,
        );

        drop(compute_pass);
        drop(render_pass);
        self.queue.submit(std::iter::once(compute_encoder.finish()));
        self.queue.submit(std::iter::once(render_encoder.finish()));

        output.present();

        Ok(())
    }
}

fn configure_surface(
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    size: (u32, u32),
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
        width: size.0,
        height: size.1,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(device, &config);
    config
}
