This README provides an overview of the approaches team *Sigmoid Sniffers* undertook to tackle the Locusta challenge. It outlines the various solution strategies we explored, describes our proxy evaluation techniques, and presents the scores achieved by each approach.
## Overview

The Locusta task involves poisoning a CIFAR-3 model by adding 10% additional training data. The challenge lies in crafting this additional data such that it degrades the performance of an unseen target model.

The evaluation model used in the RAID framework is unknown, likely a small CNN or ResNet (which later turned out to be a wrong assumption). Since the solution scores using the RAID framework were highly volatile, we eventually used a SmallCNN model (implemented in `models/model.py`) as our local proxy for more consistent evaluation. Although this didn’t eliminate score volatility, it allowed us to run hundreds of model training runs in a reasonable timeframe and helped us identify more robust trends in solution performance.

A further constraint we faced was limited compute power - Fabian did not have access to a GPU most of the time, making gradient-based adversarial image generation prohibitively slow on CPU.

## Directory Structure

The core parts of the gradient-based implementation were copied from [this repository](https://github.com/cheese-hub/SVM-Poisoning) on SVM poisoning for the MNIST dataset. First, we modified it to work with our custom CIFAR-3 data (`CIFAR Poisoning SVC.ipynb`) and later extended the approach to a CNN using back gradient descent (`CIFAR poisoning CNN.ipynb`).
Finally, more advanced label flipping approaches are detailed in `Optimized Label Flipping.ipynb`.

# 1. First Approaches: Gradient-Free Training Data Generation

Our early goal was to get working submissions on the leaderboard quickly. These first methods avoided gradients entirely and provided valuable baseline performance.

### Adding Flipped Training Data

Surprisingly, these simple label-flipping strategies performed quite well:

- **Naïve flipping:** We duplicated 250 samples of each class and reassigned them to every other class.

- **Targeted flipping:** We focused on flipping the most confusing classes (cat → horse, horse → cat).

- **Single-direction flipping:** The most effective method was to flip only one class to another (cat → horse).

The best score of 0.093 was achieved by simply adding 1500 images flipping cat to horse. Despite extensive experimentation, no other label-flipping approach outperformed this simple tactic - even on the RAID framework.
Interestingly, our proxy model (SmallCNN) later suggested that flipping all classes might have been the better approach overall, but the RAID score never confirmed this, even after numerous submissions.

### Label Flipping + Input Noise

In this variant, we added random noise to the flipped images in an attempt to further confuse the model. However, this approach consistently underperformed compared to pure label flipping.

# 2. Gradient-Based Training Data Generation

After achieving reasonably good results with simple, gradient-free methods, we explored whether we could make our adversarial data even more effective by optimizing it using gradient information.
During the Adversarial Machine Learning lecture, we studied the paper *"Poisoning Attacks against Support Vector Machines"* by Biggio et al., which presents a gradient-based data poisoning technique designed to maximally degrade model performance without regard for detectability, fitting well with our objective.

We found an existing [GitHub implementation](https://github.com/cheese-hub/SVM-Poisoning) of this approach. However, it was built specifically for the MNIST dataset. To apply it to our task, we adapted the code to support the CIFAR-3 dataset and used it to generate adversarial horse images, starting from cat images, since this direction had previously proven most effective in our label-flipping experiments.
Unfortunately, this SVM-based approach did not outperform our simple label-flipping baseline, not even on our locally trained SVM. We suspected the main issue being that SVMs are simply not suited for image classification, where deep learning models tend to dominate.

### Back-Gradient Optimization for CNNs

Based on this hypothesis, we shifted our focus to poisoning attacks tailored to deep networks. We implemented the method from *"Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization"* by Muñoz-González et al. This approach uses back gradient descent to craft poisoning samples for CNNs, which seemed more aligned with the architecture of the unknown evaluation model.
Since there was no public implementation for this technique, we developed it from scratch for CIFAR-3. As in earlier approaches, we focused on flipping class 2 to class 3 (cat → horse), since this transformation had yielded the strongest results in simpler methods.

However, despite the theoretical promise, our implementation again failed to surpass the performance of the basic label-flipping baseline on both the local CNN and for the RAID framework. This left us with several possible explanations:
- **Insufficient training data for effective gradient signals:** CIFAR-3 only has 15,000 training samples compared to 60,000 in MNIST, which may limit the model’s ability to learn the task well enough for useful poisoning gradients to emerge.
- **Low poison-to-clean ratio:** It is unclear whether 10% poisoned data is enough for gradient-based poisoning to be effective in CNNs trained on more complex data like CIFAR-3. Some prior work assumes access to larger relative poisoning budgets.
- **Limited hyperparameter tuning:** Due to time and compute constraints, we could not tune hyperparameters of the data generation at all (e.g. learning rate, poison update step size, inner loop iterations), which are known to significantly affect bilevel optimization performance.
- **Possible implementation issues:** Although we reviewed our code, bugs or numerical errors in the optimization logic may still have been present and impacted the results.

Given the time and compute constraints, we couldn’t fully explore or validate these hypotheses. Future work could involve further debugging, scaling the approach with more poison samples, or experimenting on datasets of intermediate complexity.

# 3. Last Straw: Further Optimizing Label-Flipping Approaches

As a final effort to improve our submission scores, we revisited and refined our most promising strategy: **label flipping**. This time, we aimed to make more informed decisions through systematic evaluation, training the `SmallCNN` model from our earlier implementation up to 500 times for each configuration using shuffled training data. Each experiment poisoned 10% of the training set. This large number of runs allowed us to compute stable metrics.

### First Observations: Comparing Basic Label-Flipping Strategies

We first evaluated how different flipping strategies affected the model's accuracy. Since our goal was to maximize damage, *lower accuracy means better poisoning*.

| **Method**                                           | **Mean Accuracy** | **5th Percentile** | **1st Percentile** |
|------------------------------------------------------|-------------------|--------------------|--------------------|
| Original (no poisoning)                              | 77.52%            | 73.23%             | 68.65%             |
| Flipping *all classes to all others*                 | **74.68%**        | **70.48%**         | **68.23%**         |
| Flipping only *most confusing classes*<br>(horse → cat, cat → horse) | 75.72%            | 70.64%             | 68.54%             |
| Flipping only *cat → horse*                          | 77.06%            | 72.37%             | 69.36%             |

These results suggest that **flipping all classes to all other classes** should theoretically be the most effective strategy even in the RAID framework. However, we were never able to replicate its success there. Our best RAID score of **0.093** was achieved by flipping only *cat → horse*, which performs much worse on our local predictor. 

One possibility is that we simply got lucky with that particular submission, given the **high volatility** of RAID scores. This uncertainty made further benchmarking and optimization difficult.

### Updated Image Sampling

Instead of randomly sampling images for poisoning - Why not choose the images the model already finds most confusing? To test this idea, we selected images based on how our proxy model (SmallCNN) reacted to them. We computed:

- **Confidence scores**: Images with the lowest softmax confidence were considered most ambiguous.
- **Loss values**: Images with the highest loss during training were assumed to be hardest to classify correctly.

We then selected the most confusing samples (while retaining flipping balance) based on these criteria.

> ⚠️ I sadly do not have access to the exact numerical results any more, but what i can confirm is that this approach again did not improve our performance.  

### Cheating: Using the CIFAR Test Data

One thing that actually increased our performance compared to the *cat → horse* label flipping was using the publicly available test data for our CIFAR classes instead of training datapoints.
However, since our score for a similar approach of the Cheetah task was quickly removed, we refrained from submitting such a solution.

# Conclusion

Despite testing advanced methods like gradient-based optimization and targeted label flipping, our most effective strategy remained a simple *cat → horse* label flip. Limited compute and volatile RAID scores made tuning and evaluation difficult, but extensive local testing with `SmallCNN` provided clearer insights. Due to the high variance of the RAID framework, even slightly better solutions often required many submissions to achieve a better score, if they improved it at all, making consistent progress both slow and uncertain.

# Contributions

The approaches and strategies described in this project were discussed collaboratively in team meetings.
All implementation, paper reviewing, and literature searching were carried out by Fabian Beez.
