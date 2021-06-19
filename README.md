# De-Animation
Implementation of "Selectively De-Animating Video" from SIGGRAPH 2012

## Results
 Average image of video in selected regions before and after de-animation <br><br>
<img src="imgs/guitar_average_before.jpg" height=250>&nbsp;&nbsp;&nbsp;<img src="imgs/guitar_average_after.jpg" height=250> <br>
<img src="imgs/model_average_before.jpg" height=250>&nbsp;&nbsp;&nbsp;<img src="imgs/model_average_after.jpg" height=250>


## Requirements
This code was developed on Python 3.7. Run `pip3 install -r requirements.txt` to install all dependencies

## Approach
1. Detect and track "anchor" features using Kanade-Lukas-Tomasi feature tracker and `Tracker` class defined in `klt.py`
2. Define 64 x 32 rectilinear mesh over each frame and represent each feature location as a bilinear interpolation of the vertices that make up the mesh quad which encloses the feature
3. Set up and solve for initial warp, following equation [3] in Bai et. al, using `LeastSquaresSolver` class defined in `deanimate.py`
4. Texture map input video onto initial output mesh
5. Use initial warp as input and detect "floating" feature tracks 
6. Solve for refined warp, following Equation [5] in Bai et. al
7. Texture map input video onto refined mesh

## De-Animate Your Own Videos
1. Modify `inputPath` in `config.yaml` to the path to your input video. Modify `outputDir` to your desired output directory. Set `maxFrames` to number of frames you want to process or -1 to process entire video.
2. Run `python3 deanimate.py --config config.yaml`
3. When GUI opens, select green color, adjust pen thickness, and draw de-animation strokes on image. Once complete, press Export Image button.
4. Wait for de-animation to finish and see results in `outputDir`

## Evaluation
In order to generate the average image for a de-animated video and its RGB pixel variances within the de-animation strokes, run `python3 eval.py --config config.yaml` from the command line to obtain the final results.

## References
- [Bai, Jiamin, et al. "Selectively de-animating video." ACM Trans. Graph. 31.4 (2012): 66-1.](http://graphics.berkeley.edu/papers/Bai-SDV-2012-08/Bai-SDV-2012-08_large.pdf)
- [Custom Tkinter Button](https://github.com/TomSchimansky/GuitarTuner)
