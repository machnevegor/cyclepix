# CyclePix

Implement and evaluate a CycleGAN model for image-to-image translation using the WikiArt dataset, enabling
transformation between different artistic styles without paired datasets.

## Team

| Name                   | Innomail                          |
|------------------------|-----------------------------------|
| Egor Machnev           | e.machnev@innopolis.university    |
| Apollinaria Chernikova | a.chernikova@innopolis.university |

## Project Overview

Recent advances in Generative Adversarial Networks (GANs) have significantly improved the ability to translate images
between domains without paired datasets. In this project, we propose implementing CycleGAN for image-to-image
translation, leveraging the WikiArt dataset from Hugging Face to explore different painting styles. Our goal is to
develop a model that can convert images from one artistic style to another without needing corresponding image pairs.
CycleGAN’s cycle-consistency loss ensures that an image translated from domain A to domain B can be mapped back to its
original form, preserving essential content. This approach is useful in artistic style transfer, domain adaptation, and
data augmentation.
