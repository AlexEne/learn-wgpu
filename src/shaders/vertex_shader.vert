#version 450

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float tu, tv;
};

layout(std430, set = 0, binding = 0) readonly buffer GeometryData {
    Vertex vertices[];
};

layout(location = 0) out vec2 v_tex_coords;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_world_position;
layout(location = 3) out vec3 v_camera_pos;

layout(set = 1, binding = 0) uniform Camera {
    mat4 camera_mat;
    vec4 cam_pos;
};

layout(location = 5) in vec4 x_axis;
layout(location = 6) in vec4 y_axis;
layout(location = 7) in vec4 z_axis;
layout(location = 8) in vec4 w_axis;

void main()
{
    mat4 object_mtx = mat4(x_axis, y_axis, z_axis, w_axis);
    
    Vertex vertex = vertices[gl_VertexIndex];
    vec4 position = vec4(vertex.px, vertex.py, vertex.pz, 1.0);
    vec4 normal = vec4(vertex.nx, vertex.ny, vertex.nz, 1.0);
    vec2 tex_coords = vec2(vertex.tu, vertex.tv);
    
    vec4 pos = object_mtx * position;
    v_world_position = pos.xyz;
    // v_position.z = -v_position.z;
    gl_Position = camera_mat * pos;
    v_tex_coords = tex_coords;
    v_normal = normalize((object_mtx * normal).xyz);
    v_camera_pos = cam_pos.xyz;
}

