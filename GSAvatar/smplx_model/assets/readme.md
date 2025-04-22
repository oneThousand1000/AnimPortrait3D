### `lower_teeth.obj` and `upper_teeth.obj`  
Sourced from [TurboSquid](https://www.turbosquid.com/3d-models/realistic-human-jaws-and-tongue-3d-model-2014042) under the [Standard License](https://blog.turbosquid.com/turbosquid-3d-model-license/).  
The tongue component has been excluded, retaining only the detailed teeth mesh to improve segmentation accuracy.  

### `l_eyelid.npy` and `r_eyelid.npy`  
Sourced from [metrical-tracker](https://github.com/Zielon/metrical-tracker/tree/master/flame/blendshapes). These blendshapes control the opening and closing of the eyelids.  

### `head_template_mesh.obj`  
Mean **SMPL-X** mesh.  

### `flame_landmark_embedding_with_eyes.npy`  
Sourced from [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars).  

### `flame_landmark_embedding_with_eyes_in_smplx_order.npy`  
A re-ordered version of `flame_landmark_embedding_with_eyes.npy` to match the **SMPL-X** model format.  

### `FLAME_masks_v2.pkl`  
Derived from `FLAME_masks.pkl`, with additional mask types added. See the `create_custom_mask` function in `smplx.py` for details.  

### `inner_eye_to_be_removed.txt`  
Specifies redundant inner eye components that will be removed.  

### `lower_teeth_ids.npy`  
Vertex IDs of the lower teeth, excluding gums.  

### `upper_teeth_ids.npy`  
Vertex IDs of the upper teeth, excluding gums.  

### `seg_in_cropped_mesh.pkl`  
Precomputed face IDs for different regions in the cropped mesh.  

### `smplx_mean_lm3d.npy`  
Global landmark positions of the mean **SMPL-X** model.  

### `smplx_vert_segmentation.json`  
Sourced from [Meshcapade](https://meshcapade.wiki/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json), containing vertex segmentation for **SMPL-X**.  

 