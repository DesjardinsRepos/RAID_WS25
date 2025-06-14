# Snicket

## Contributors

* Fabian: Implemented all Solutions

## Solutions

All solutions (except the first) are machine-learning based and implemented in `snicket/solutions.ipynb`.
A short summary of them is also provided in our presentation at `snicket/slides.pptx`.

### Solution 1a

Simply `return false`. Score: 50%

### Solution 1b: ResNet

Intuitively, we thought that detecting the humanly visible artifacts in the midjourney images could lead to a good solution and implemented a CNN approach.
As a baseline, we implemented a simple ResNet. 5 Epochs, HorizontalFlip augmentation, ImageNet Weights initially, Adam + cross entropy. Score: 88.7%

Updating this architecture with more epochs, `ReduceLROnPlateau`, uneven loss weights and ImageNet-inspired data augmentation got our score to 91.3%.

### Solution 1c: EfficientNet

Inspired by regular image classification, we implemented an EfficientNet-BO for this task and got 95.4% with the same setup as the previous ResNet.
Adding label smoothing while lowering dropout, manually optimizing the main hyperparameters and completely removing data augmentation further increased our score to 97.1%.

**An interesting learning**: Since ImageNet-inspired data augmentation actually decreases performance and augmentation functions like crop, flip and colorjitter do not reduce the visibility of our previous encountered humanly visible artifacts, the network seems to be learning something else entirely.
Probably some artifacts that we as a human cannot see intuitively.

Since we now have only 7000+1000 train/val datapoints, we tried to find additional training data online. 
We found the GenImage dataset and one named CIFAR-inspired.
We chose to try the latter since it seemed like its midjourney version is closer to our training data (and probably to our test data too, since we are already getting such a high accuracy).
However, including this data in the train set actually decreased our performance to 96.2%.

Since we were already using heavy regularization (dropout, label smoothing, weight decay), we did not think that increasing model capacity would increase performance.
So as we tested an EfficientNet-B1, we were pretty suprised that it actually increased our performance by roughtly 50% to 98.2%.
Sadly, scaling to B2 actually overfitted the network, which was probably also the reason why all EfficientNet-V2 models underperformed even an EfficientNet-B0.

Another thing we testet was feeding frequency information into our network. 
In the GenImage paper, the average DFT of noise residuals significantly differs from ImageNet to their Midjourney data, so we hoped to include this information to exploit potential differences and also make possible artifacts more visible to the network.
Sadly, this actually decreased performance.

Finally, we did a proper and time-consuming hyperparameter optimization, which earned us our badge (98.6%) and trained on train+val combined after, which further increased the score to 98.7%.
For the latter, we changed our learning rate scheduling from `ReduceLROnPlateau` to `CosineAnnealingWarmRestarts` since a) we did not have the validation data any more and b) cosine annealing is closer to the state of the art, while taking a longer time to train the network.
However, performance was no issue since we already optimized our hyperparameters.

Naturally, we testet much more minimal changes in addition to the mentioned points, but to keep this short i only included what seemed interesting.

---

# Vader

## Contributors

* Fabian: Implemented Solutions 2a and 2b. Experimented with MOG2 and SelfieSegmentation masks, which however could not increase the score.
* ...

## Solutions

### Solution 2a

First, we took the average pixel value as a solution in `vader/0.014.ipynb`. Score: 0.014

### Solution 2b

What significantly improved our score was taking the second most frequent pixel value instead in `vader/0.027.ipynb`. Score: 0.027

---

# Animagus

## Contributors

* Fabian: Implemented Solution 3a
* ...

## Solutions

### Solution 3a

First, we simply encoded the length of the encoded bytes in the first byte and the encoded bytes after in `animagus/0.1563.py`. Score: 0.156
