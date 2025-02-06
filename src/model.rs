use crate::{
    pipelines::{MaterialData, TextureID},
    renderer,
    texture::Texture,
};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

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
    pub bounding_sphere: BoundingSphere,
}

impl Model {
    pub fn from_gltf(
        renderer: &renderer::Renderer,
        path: &str,
        textures: &mut Vec<Texture>,
    ) -> Vec<Model> {
        let (document, buffers, _images) = gltf::import(path).unwrap();

        let mut texture_map = HashMap::new(); // Map gltf ids to texture ids
        let mut models = Vec::new();

        for mesh in document.meshes() {
            for (idx, primitive) in mesh.primitives().enumerate() {
                let mut vertices = vec![];
                let mut min_vtx_pos = Vec3::ZERO;
                let mut max_vtx_pos = Vec3::ZERO;

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();
                let Some(tex_coords) = reader.read_tex_coords(0) else {
                    println!(
                        "No tex_coords found for mesh: {:?} primitive {:?}",
                        mesh.name(),
                        idx
                    );
                    continue;
                };

                let tex_coords: Vec<[f32; 2]> = tex_coords.into_f32().collect();

                let normals = reader.read_normals().unwrap();

                for ((position, tex_coord), normal) in positions.zip(tex_coords).zip(normals) {
                    // Adjust AABB
                    for i in 0..3 {
                        if position[i] < min_vtx_pos[i] {
                            min_vtx_pos[i] = position[i];
                        }

                        if position[i] > max_vtx_pos[i] {
                            max_vtx_pos[i] = position[i];
                        }
                    }

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
                let Some(base_texture) = pbr_metalic_roughness.base_color_texture() else {
                    println!(
                        "No base_color_texture found for mesh: {:?} primitive {:?}",
                        mesh.name(),
                        idx
                    );
                    continue;
                };

                let base_texture_id = base_texture.texture().source().index();

                let base_color_texture =
                    if let Some(base_color_texture_id) = texture_map.get(&base_texture_id) {
                        *base_color_texture_id
                    } else {
                        let base_color_texture = get_texture(&buffers, base_texture);
                        let texture = Texture::from_bytes(
                            renderer,
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

                let Some(metalid_roughness_texture_id) =
                    pbr_metalic_roughness.metallic_roughness_texture()
                else {
                    println!(
                        "No metallic_roughness_texture found for mesh: {:?} primitive {:?}",
                        mesh.name(),
                        idx
                    );
                    continue;
                };

                let metalid_roughness_texture_id =
                    metalid_roughness_texture_id.texture().source().index();

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
                        renderer,
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

                let sphere_center = (min_vtx_pos + max_vtx_pos) / 2.0;
                let sphere_radius = (max_vtx_pos - min_vtx_pos).length() / 2.0;

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
                    bounding_sphere: BoundingSphere {
                        center: [sphere_center.x, sphere_center.y, sphere_center.z],
                        radius: sphere_radius,
                    },
                });
            }
        }

        models
    }

    fn create_buffers(
        &self,
        renderer: &renderer::Renderer,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let vertex_buffer = renderer.create_buffer_init(
            "Vertex Buffer",
            bytemuck::cast_slice(&self.vertices),
            wgpu::BufferUsages::STORAGE,
        );

        let index_buffer = renderer.create_buffer_init(
            "Index Buffer",
            bytemuck::cast_slice(&self.indices),
            wgpu::BufferUsages::INDEX,
        );

        let bounding_shpere = renderer.create_buffer_init(
            "AABB Buffer",
            bytemuck::cast_slice(&[self.bounding_sphere]),
            wgpu::BufferUsages::UNIFORM,
        );

        (vertex_buffer, index_buffer, bounding_shpere)
    }

    pub fn create_gpu_data(&self, renderer: &renderer::Renderer) -> ModelGPUData {
        let (vertex_buffer, index_buffer, bounding_sphere) = self.create_buffers(renderer);

        ModelGPUData {
            vertex_buffer,
            index_buffer: WGPUIndexBufferData {
                index_buffer,
                num_indices: self.indices.len() as _,
                index_buffer_format: wgpu::IndexFormat::Uint32, // Wasteful for now
            },
            bounding_sphere,
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

#[repr(C)]
#[derive(Debug, Copy, Default, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundingSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

pub struct ModelGPUData {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: WGPUIndexBufferData,
    pub bounding_sphere: wgpu::Buffer,
}

pub struct ModelGPUDataInstanced {
    pub model_gpu_data: ModelGPUData,
    pub instance_buffer: wgpu::Buffer,
    pub instance_output_buffer: wgpu::Buffer,
    pub num_instances: u32,
}

pub struct WGPUIndexBufferData {
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub index_buffer_format: wgpu::IndexFormat,
}
