use glam::Vec3;

pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub up: Vec3,

    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.position, self.position + self.direction, self.up);
        let projection =
            glam::Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far);
        projection * view
    }
}
