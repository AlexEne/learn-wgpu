#version 450

// layout(set = 0, binding = 0) uniforom vec3 color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(1.0);
}