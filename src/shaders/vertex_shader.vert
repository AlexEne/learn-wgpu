#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coords;

layout(location = 0) out vec2 v_tex_coords;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_position;

layout(set = 1, binding = 0) uniform Camera {
    mat4 camera_mat;
};

layout(location = 5) in vec4 x_axis;
layout(location = 6) in vec4 y_axis;
layout(location = 7) in vec4 z_axis;
layout(location = 8) in vec4 w_axis;

void main()
{
    mat4 object_mtx = mat4(x_axis, y_axis, z_axis, w_axis);
    vec4 pos = object_mtx * vec4(position, 1.0);
    v_position = pos.xyz;
    gl_Position = camera_mat * pos;
    v_tex_coords = tex_coords;
    v_normal = normalize((object_mtx * vec4(normal, 1.0)).xyz);
}

