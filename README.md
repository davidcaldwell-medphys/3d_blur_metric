# 3d_blur_metric
3D Blur Metric (BM3D) for coronary angiography sequences.

Extends the 2D no-reference blur metric of Crete-Roffet et al. (2007) into the
spatio-temporal domain. Coronary angiography sequences form a (T, H, W) volume;
ghosting and lag artefacts introduced by video frame interpolation algorithms
manifest as blurring in the temporal (t) direction, which standard 2D metrics
do not capture. BM3D quantifies blurring across all three axes simultaneously.

BM3D ranges from 0 (no blur, all edge detail preserved) to 1 (fully blurred).

At lower frame rates, consecutive frames are further apart in time, so the same
physical motion produces larger inter-frame pixel differences. Without any
correction, a 7.5 FPS acquisition would appear artificially sharper in the
temporal direction than an equivalent 15 FPS acquisition simply because of the
larger frame spacing. To account for this, temporal differences are scaled by
fr_ref / fr, where fr_ref is the highest frame rate in the study (15 FPS here).

Reference:
    Crete-Roffet F, Dolmiere T, Ladret P, Nicolas M (2007). The blur effect:
    perception and estimation with a new no-reference perceptual blur metric.
    Proc. SPIE 6492, Human Vision and Electronic Imaging XII.
