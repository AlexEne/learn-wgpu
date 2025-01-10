import os
import subprocess

# glslc = "./tools/glslc.exe"
glslc = "./tools/glslang.exe"

def build_shaders_in_folder(folder_path, output_folder):
    # Get all GLSL files in the folder
    print(f'Building shaders in {folder_path} to {output_folder}\n')
    
    # Find vertex and fragment shader files separately
    vertex_shader_files = [f for f in os.listdir(folder_path) if f.endswith('.vert')]
    fragment_shader_files = [f for f in os.listdir(folder_path) if f.endswith('.frag')]

    # Process vertex shaders
    for vertex_shader in vertex_shader_files:
        vertex_shader_path = os.path.join(folder_path, vertex_shader)
        vertex_shader_output = os.path.join(output_folder, vertex_shader.replace('.vert', '_vs.spv'))
        
        # Build vertex shader
        result = subprocess.run([glslc, vertex_shader_path, '-g', '-I{folder_path}', '-V', '-o', vertex_shader_output], capture_output=True, text=True)
        print(f'Building {vertex_shader} as vertex shader')
        print(result.stdout)
        print(result.stderr)
        
    # Process fragment shaders
    for fragment_shader in fragment_shader_files:
        fragment_shader_path = os.path.join(folder_path, fragment_shader)
        fragment_shader_output = os.path.join(output_folder, fragment_shader.replace('.frag', '_fs.spv'))
        
        # Build fragment shader
        result = subprocess.run([glslc, fragment_shader_path, '-g', '-I{folder_path}', '-V', '-o', fragment_shader_output], capture_output=True, text=True)
        print(f'Building {fragment_shader} as fragment shader')
        print(result.stdout)
        print(result.stderr)


if __name__ == "__main__":
    build_shaders_in_folder('src/shaders', output_folder='data/spirv')