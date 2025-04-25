

## Q&A


> ImportError: ('Unable to load OpenGL library', "/home/xxx/envs/AnimPortrait3D/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-13.so.1)", 'libOSMesa.so.8', 'libOSMesa.so.8')

Solution: https://github.com/pybind/pybind11/discussions/3453#discussioncomment-1926590

 

 
> How to select "abstract prompt"?


Solution: For the mouth and eye regions, which generally lack person-specific features, we observe that providing overly detailed prompts to the ControlNet can degrade image quality. Therefore, we use more abstract text prompts combined with region-specific prefixes to guide these areas, broadly categorizing the avatar. In practice, you can simply input a very rough prompt that describes the avatar in just a few words, typically indicating only its gender and age (e.g., "a boy", "an old man", "a woman").