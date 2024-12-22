# Learning WGPU

I made this repo because I'm also trying to learn how to organize rendering engine to suit my needs (for my isometric game https://dwarf.world).  
I don't need much for my game, but I want like to be able to have some nice lighting, shadows, fog, snow/rain effects while keeping the pixelated look.
I also am in the process of transitioning from 2D quads of isometric art to 3D as, believe it or not, it makes lots of things easier (shadows for example).
The art style will stay the same, just with nicer and easier to implement effects.

For context, the only graphics API i'm somewhat familiar with was OpenGL.  
In terms of graphics rendering techniques my knowledge is also super basic.
I can render a textured cube, I know how transforms and shaders work at a basic level, so, just the absolute basics.  

While I could implement these in OpenGL, I'm doing this transition because OpenGL is a pain to work with today.  
Debug support is non-existent and there are alternatives with better tooling available out there (WGPU, DirectX11/12, Metal, Vulkan).  
I landed on wgpu as my graphics abstraction layer because it works on all platforms (including web), has nice tooling available (e.g. I can debug shaders in Renderdoc and xcode) and is easier to learn compared to Vulkan.

So, this repo is me mostly learning: 
1) how to work with modern graphics APIs
2) trying to organise a renderer so it's not a complete mess
3) Some graphics techniques.


I've mostly worked through this tutorial: https://sotrh.github.io/learn-wgpu/
I didn't follow it fully and towards the last commits, as you'll see, I started going more towards loading gLTF files and working on rendering them.

There are some high-quality modern renderers out there. The best one that springs to mind is: https://github.com/EmbarkStudios/kajiya (on top of Vulkan).
But I found it overwhelming for me to try to understand the reasons behind its design.  
It's because it combined a bunch of things I know nothing about: modern rendering techniques like raytracing, etc., modern graphics APIs like Vulkan and modern renderer architectures based on render graphs.


Meanwhile, I had/have trouble answering really basic rendering engine questions like:   
What's a material?! Is it a shader?! Is it shader + textures and other properties?!  
Should a Material then own the Textures it needs for an object?  
Should the Mesh struct own the material?  

I don't know yet the answer to most of those questions, and what I'm doing here is just exploring a bit various kinds of architectures from first principles:
Is it simple to use and understand (for my needs)?  
Does it have some clear ownership?  
Is it flexible and easy to extend in the right places?  

In the meantime, here's some Avocados rendered with it:
<img width="799" alt="Screenshot 2024-12-22 at 16 20 58" src="https://github.com/user-attachments/assets/aec17b46-d23e-459e-937a-88d887e6c27c" />


