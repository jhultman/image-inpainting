## Image inpainting
Here we use convex optimization to inpaint missing pixels in corrupted images. We find the discrete gradient and minimize its l1 norm under the constraint that the recovered image must match the corrupt image at those indices where the pixels are valid. I first saw this problem in a [lecture](https://www.youtube.com/watch?v=C7gZzhs6JMk) by Stephen Boyd and Steven Diamond (Stanford).

### Examples
1. Text removal.
![Text](./images/readme/mona_lisa_text_results.png)

2. Extreme sparsity (90% missing data).
![Noise](./images/readme/mona_lisa_noisy_results.png)

3. Using crude heuristic for identifying bad pixels.
![Watermark](./images/readme/watermark_results.png)
