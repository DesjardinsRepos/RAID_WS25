# Snicket

## Contributors

* Fabian: Implemented all Solutions

## Solutions

All solutions are machine-learning based and implemented in `snicket/snicket.ipynb`.
A short summary of them is also provided in our presentation at `snicket/slides.pptx`.
The files starting with "submit" are the server submission workflows for every model, and there are some bash files for an automated submission.
`grad-cam.ipynb` visualizes gradient information and there are some python files for hyperparameter optimization too.

### Solution A

Simply `return false`. Score: 50%

### Solution B: ResNet

Intuitively, we thought that detecting the humanly visible artifacts in the midjourney images could lead to a good solution and implemented a CNN approach.
As a baseline, we implemented a simple ResNet. 5 Epochs, HorizontalFlip augmentation, ImageNet Weights initially, Adam + cross entropy. Score: 88.7%

Updating this architecture with more epochs, `ReduceLROnPlateau`, uneven loss weights and ImageNet-inspired data augmentation got our score to 91.3%.

### Solution C: EfficientNet

Inspired by regular image classification, we implemented an EfficientNet-BO for this task and got 95.4% with the same setup as the previous ResNet.
Adding label smoothing while lowering dropout, manually optimizing the main hyperparameters and completely removing data augmentation further increased our score to 97.1%.

**An interesting learning**: Since ImageNet-inspired data augmentation actually decreases performance and augmentation functions like crop, flip and colorjitter do not reduce the visibility of our previous encountered humanly visible artifacts, the network seems to be learning something else than these artifacts too.
Probably something that we as a human cannot see intuitively.

Since we now have only 7000+1000 train/val datapoints, we tried to find additional training data online. 
We found the GenImage dataset and one named [CIFAKE-inspired](https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired).
We chose to try the latter since it seemed like its midjourney version is closer to our training data (and probably to our test data too, since we are already getting such a high accuracy).
However, including this data in the train set actually decreased our performance to 96.2%.

Since we were already using heavy regularization (dropout, label smoothing, weight decay), we did not think that increasing model capacity would increase performance.
So as we tested an EfficientNet-B1, we were pretty suprised that it actually increased our performance by roughtly 50% to 98.2%.
Sadly, scaling to B2 actually overfitted the network, which was probably also the reason why all EfficientNet-V2 models underperformed even an EfficientNet-B0.

Another thing we tested was feeding frequency information into our network as a 4th channel. 
In the [GenImage paper](https://arxiv.org/pdf/2306.08571), the average DFT of noise residuals significantly differs from ImageNet to their Midjourney data, so we hoped to include this information to exploit potential differences and also make possible artifacts more visible to the network.
Sadly, this actually decreased performance.

Finally, we did a proper and time-consuming hyperparameter optimization, which earned us our badge (98.6%) and trained on train+val combined after, which further increased the score to 98.7%.
For the latter, we changed our learning rate scheduling from `ReduceLROnPlateau` to `CosineAnnealingWarmRestarts` since a) we did not have the validation data any more and b) cosine annealing is closer to the state of the art, while taking a longer time to train the network.
However, performance was no issue since we already optimized our hyperparameters.

Naturally, we testet much more minimal changes in addition to the mentioned points, but to keep this short i only included what seemed interesting.

---

# Vader

## Contributors

Briefly describe who contributed to this challenge and how.

* Fabian Beez: Proposed the idea for A and implemented A. Experimented with MOG2 and SelfieSegmentation masks, which however could not increase the score.
* Lars Böke: Proposed the idea for B, C, D and implemented B, C, D.

## Solutions

### Solution A

As an initial and straightforward approach to the problem, we used calculated the average pixel value and then the second most frequent pixel value. Both approaches can be found in Vader/inital_attempt.py. They resulted in a score of 0.014 and 0.027 respectively.

### Solution B

Building on the previous approach we implemented basic_solution.py. It is still a straight forwards solution. First, the virtual background is estimated by the median over all frames and then all the pixels that are different enough (surpass a threshold) are counted as leaks. The reconstruction of the real background is done with the mean of the leaked pixels.

### Solution C

To improve we investigated prior work and found the paper Background Buster by Mohd et al. (https://ieeexplore.ieee.org/document/9833657). This paper does not provide code, but felt like a more sophisticated version of our solution B and the steps were reasonably well explained. We implemented background_buster.py

The core idea is to mask everything in he videos frames but the leaks. Therefore, we first estimate the virtual background by stability, and then we create a virtual background mask, a blending zone mask and a person mask. The resulting pixels are the leaks which are then aggregated by there mean value.

Solution B and C were not submitted to the server as we encountered session problems (I only learned about tmux during solution D) and dependency issues we only resolved when already working on solution D.

### Solution D

In our last solution background_buster_imporved.py we tried to improve our solution C. Frist we removed the blending mask, as it never resulted in better scores. The paper experimented with relatively stationary users with is a big difference to our use case.  
We then estimated the vb by using the most frequent occurring Color (mode) and we used DeepLabV3 for the person mask (DeepLabV3 is very resource intensive but outperformed all other approaches including moc2 by a lot so we stuck to it). With is we achieved a local score of 0.462 and a server score of 0.0602.

The local score is the score of evaluating the code with one video.

Then then added an adaptive virtual background mask and achieved a local score of 0.4722 and a server score of 0.062.
Lastly, we changed our background reconstruction to prioritise pixels that are different to the virtual background, as we noticed that a lot of leaks have overlapping real background and virtual background and we added an inpaint function that would guess the value of black pixels. This approach lead to a local score of 0.564.

Sadly, we were not able to submit the last modification of solution D to the server. Our algorithms took 7-8h to complete the tests even before the changes, and after the code got stuck or failed 3 times, we did not have time to submit it again. 


---

# Animagus

## Contributors

Briefly describe who contributed to this challenge and how.

* Fabian Beez: Proposed the idea for A and implemented A.
* Lennart Dammer: Proposed the idea for B,C,D and implemented B,C,D.

## Solutions

### Solution A

As an initial and straightforward approach, we simply overwrite the beginning of the carrier file with the message bytes.
This approach is implemented in `encoder/overwrite.py` and `decoder/overwrite.py`

### Solution B

To avoid direct overwriting, we explored encoding data into the structure of the text itself. We initially chose to encode the message via character case modification (flipping uppercase/lowercase to represent bits), as this approach seemed most promising in terms of offering enough modification points to embed the full message. Alternatives like using whitespace or newline characters were also considered, but we deprioritized them early on, as their encoding capacity appeared uncertain and potentially too limited depending on the structure of the carrier file.
However, the resulting file had a large edit distance from the original, making it unsuitable for use under the challenge constraints.
This approach is implemented in `encoder/case_mod.py` and `decoder/case_mod.py`

### Solution C

This solution builds upon Solution B's case-modification approach but introduces an optimization technique to minimize the number of case flips needed to encode the payload. The core concept is inspired by barcode check digits and matrix-based encoding.

How It Works:

The approach is similar to how barcode check digits work. For example:
- Consider a barcode with 3 numbers and one check digit
- If our carrier numbers are *4*72, the check digit is calculated as: (1*4 + 2*7 + 3*2) % 11 = 2
- To store a different value (e.g., "3"), we can modify the carrier to *5*72: (1*5 + 2*7 + 3*2) % 11 = 3

The Concept of Our Method:
Instead of using simple modulo calculations with numbers, our solution uses a much more sophisticated method:

We use a pseudo-randomly generated matrix G of size "characters in carrier × bits in message" and calculate G * c where c is a vector of the carrier bits. The result is the natural encoded message. We use a clever method to modify c into c' such that G * c' = m, our wanted message, with very few bit flips.

This appoach works technically:

On: carrier_files/cophercommon_gcm-18.h  
Message: b'What a great day!'  
Message length: 17 bytes  
**Solution B:** flips: 54  
**Solution C:** flips: 37


However the resultes not good (Score 0.1522 (Edit-Distance: 0.0607 , Perplexity: 29.7765)). While the method is conceptually interesting, it performs poorly under the evaluation criteria used in this challenge.

This approach is implemented in `encoder/case_mod_min.py` and `decoder/case_mod_min.py`

### Solution D

We returned to the original idea of overwriting (Solution A) but tried to make space in the carrier file to reduce the amount of overwriting needed.

We investigated two preprocessing strategies:

1. Detect and exploit `\r\n` newline encodings by stripping `\n` to free up bytes — not applicable, as the files use only `\n`.
2. Remove spaces before newline characters and use the saved space to append message bytes — this worked and **is the best-performing solution overall**.

This approach is implemented in `encoder/newline_whitespaces.py` and `decoder/newline_whitespaces.py`.
