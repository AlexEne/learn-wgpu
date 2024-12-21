use gltf::mesh::util::indices;
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
    pub tex_coords: [f32; 2],
}

pub struct Model {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>, // A bit wasteful since they could be u16
}

impl Model {
    pub fn from_gltf(path: &str) -> Model {
        let (document, buffers, _images) = gltf::import(path).unwrap();
        let mut vertices = vec![];
        let mut indices = vec![];

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();
                let tex_coords: Vec<[f32; 2]> =
                    reader.read_tex_coords(0).unwrap().into_f32().collect();

                for (position, tex_coord) in positions.zip(tex_coords) {
                    let vertex = ModelVertex {
                        position,
                        tex_coords: tex_coord,
                    };
                    vertices.push(vertex);
                }

                indices = reader.read_indices().unwrap().into_u32().collect();
                break;
            }
        }

        Model { vertices, indices }
    }

    pub fn create_buffers(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
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

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

        VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}
