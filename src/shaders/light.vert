#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Camera {
    mat4 camera_mat;
};

layout(set = 1, binding = 0) uniform Obj {
    mat4 object_mtx;
};

void main()
{
    gl_Position = camera_mat * object_mtx * vec4(position, 1.0);
}