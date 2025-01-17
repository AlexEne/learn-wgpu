use crate::{
    material::{MaterialData, TextureID},
    texture::Texture,
};
use std::collections::HashMap;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    VertexAttribute, VertexBufferLayout,
};

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

pub struct Model {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>, // A bit wasteful since they could be u16
    pub material: MaterialData,
}

impl Model {
    pub fn from_gltf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        textures: &mut Vec<Texture>,
    ) -> Vec<Model> {
        let (document, buffers, _images) = gltf::import(path).unwrap();

        let mut texture_map = HashMap::new(); // Map gltf ids to texture ids
        let mut models = Vec::new();

        for mesh in document.meshes() {
            for (idx, primitive) in mesh.primitives().enumerate() {
                let mut vertices = vec![];

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();
                let tex_coords: Vec<[f32; 2]> =
                    reader.read_tex_coords(0).unwrap().into_f32().collect();

                let normals = reader.read_normals().unwrap();

                for ((position, tex_coord), normal) in positions.zip(tex_coords).zip(normals) {
                    let vertex = ModelVertex {
                        position,
                        normal,
                        tex_coords: tex_coord,
                    };
                    vertices.push(vertex);
                }

                let indices = reader.read_indices().unwrap().into_u32().collect();

                let primitive_material = primitive.material();

                let pbr_metalic_roughness = primitive_material.pbr_metallic_roughness();
                let base_texture = pbr_metalic_roughness.base_color_texture().unwrap();

                let base_texture_id = base_texture.texture().source().index();

                let base_color_texture =
                    if let Some(base_color_texture_id) = texture_map.get(&base_texture_id) {
                        *base_color_texture_id
                    } else {
                        let base_color_texture = get_texture(&buffers, base_texture);
                        let texture = Texture::from_bytes(
                            device,
                            queue,
                            wgpu::TextureFormat::Rgba8UnormSrgb,
                            &base_color_texture,
                            &format!("base_color_texture {:?}", idx),
                        )
                        .unwrap();

                        textures.push(texture);
                        let id = textures.len() - 1;
                        texture_map.insert(base_texture_id, TextureID(id));
                        TextureID(id)
                    };

                let metalid_roughness_texture_id = pbr_metalic_roughness
                    .metallic_roughness_texture()
                    .unwrap()
                    .texture()
                    .source()
                    .index();

                let metalic_roughness_texture = if let Some(metalic_roughness_texture_id) =
                    texture_map.get(&metalid_roughness_texture_id)
                {
                    *metalic_roughness_texture_id
                } else {
                    let metalic_roughness_texture = get_texture(
                        &buffers,
                        pbr_metalic_roughness.metallic_roughness_texture().unwrap(),
                    );
                    let texture = Texture::from_bytes(
                        device,
                        queue,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        &metalic_roughness_texture,
                        &format!("metallic_roughness_texture {:?}", idx),
                    )
                    .unwrap();

                    textures.push(texture);
                    let id = textures.len() - 1;
                    texture_map.insert(metalid_roughness_texture_id, TextureID(id));
                    TextureID(id)
                };

                let metalic_factor = pbr_metalic_roughness.metallic_factor();
                let roughness_factor = pbr_metalic_roughness.roughness_factor();
                let base_color_factor = pbr_metalic_roughness.base_color_factor().into();

                models.push(Model {
                    vertices,
                    indices,
                    material: MaterialData::new(
                        base_color_texture,
                        metalic_roughness_texture,
                        metalic_factor,
                        roughness_factor,
                        base_color_factor,
                    ),
                });
            }
        }

        models
    }

    fn create_buffers(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    pub fn create_gpu_data(&self, device: &wgpu::Device) -> ModelGPUData {
        let (vertex_buffer, index_buffer) = self.create_buffers(device);
        ModelGPUData {
            vertex_buffer,
            index_buffer: WGPUIndexBufferData {
                index_buffer,
                num_indices: self.indices.len() as _,
                index_buffer_format: wgpu::IndexFormat::Uint32, // Wasteful for now
            },
        }
    }
}

fn get_texture(
    buffers: &Vec<gltf::buffer::Data>,
    base_texture: gltf::texture::Info<'_>,
) -> Vec<u8> {
    let source = base_texture.texture().source().source();

    match source {
        gltf::image::Source::View { view, .. } => {
            let buffer = &*buffers[view.buffer().index()];
            let start = view.offset();
            let end = start + view.length();
            return buffer[start..end].to_vec();
        }
        _ => {
            panic!("Unsupported texture source (URI)");
        }
    }
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

        VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

pub struct ModelGPUData {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: WGPUIndexBufferData,
}

pub struct ModelGPUDataInstanced {
    pub model_gpu_data: ModelGPUData,
    pub instance_buffer: wgpu::Buffer,
    pub num_instances: u32,
}

pub struct WGPUIndexBufferData {
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub index_buffer_format: wgpu::IndexFormat,
}
