#version 450
#extension GL_GOOGLE_include_directive : enable

#include "types.glsl"

layout(location = 0) in vec2 v_tex_coords;
layout(location = 1) in vec3 v_normal;

layout(set = 0, binding = 0) uniform texture2D t_texture;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

layout(set = 2, binding = 0) uniform Light {
    vec3 position;
    vec3 color;
} light;

layout(location = 0) out vec4 out_color;

void main()
{
    out_color = texture(sampler2D(t_texture, s_diffuse), v_tex_coords);
}
