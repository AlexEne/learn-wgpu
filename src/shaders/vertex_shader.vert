#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec2 v_tex_coords;

layout(set = 1, binding = 0) uniform Camera {
    mat4 camera_mat;
};


void main()
{
    gl_Position = camera_mat * vec4(position, 1.0);
    v_tex_coords = tex_coords;
}

